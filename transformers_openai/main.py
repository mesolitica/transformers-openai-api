from transformers_openai.env import args
from transformers_openai.base_model import ChatCompletionForm
from transformers_openai.function import cleanup_cache
import asyncio
import logging
import uuid
import time
import torch
import uvicorn
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi import File, Form, UploadFile
from sse_starlette import EventSourceResponse
from transformers import cache_utils
from collections import deque

if args.serving_type == 'chat':
    if args.accelerator_type == 'cuda':
        from transformers_openai.main_cuda import (
            chat_completions,
            load_model,
            prefill,
            step,
        )
if args.serving_type == 'whisper':
    if args.accelerator_type == 'cuda':
        from transformers_openai.main_whisper_cuda import (
            audio_completions,
            load_model,
            prefill,
            step,
        )


HEADERS = {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
}


class InsertMiddleware:
    def __init__(self, app, max_concurrent=50):
        self.app = app
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue = deque()

    async def process_request(self, scope, receive, send):
        async with self.semaphore:

            log = f"Received request {scope['request']['uuid']} in queue {scope['request']['time_in_queue']} seconds"
            logging.info(log)

            scope['cache'] = getattr(cache_utils, args.cache_type, None)
            if scope['cache'] is not None:
                if args.continous_batching:
                    logging.warning('continous batching is enable, will ignore `CACHE_TYPE`.')
                    scope['cache'] = None
                else:
                    scope['cache'] = scope['cache']()

            queue = asyncio.Queue()

            async def message_poller(sentinel, handler_task):
                nonlocal queue
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        handler_task.cancel()
                        return sentinel
                    await queue.put(message)

            sentinel = object()
            handler_task = asyncio.create_task(self.app(scope, queue.get, send))
            asyncio.create_task(message_poller(sentinel, handler_task))

            try:
                await handler_task
                if 'time_max_tokens' in scope['request']:
                    time_taken_first_token = scope['request']['time_first_token'] - \
                        scope['request']['after_queue']
                    time_taken_max_tokens = scope['request']['time_max_tokens'] - \
                        scope['request']['time_first_token']
                    tps = scope['request']['total_tokens'] / time_taken_max_tokens

                    if 'total_seconds' in scope['request']:
                        sps = scope['request']['total_seconds'] / time_taken_max_tokens
                        extra_whisper = f', Seconds Per Second {sps}'
                    else:
                        extra_whisper = ''

                    s = f"Complete {scope['request']['uuid']}, time first token {time_taken_first_token} seconds, time taken {time_taken_max_tokens} seconds, TPS {tps}{extra_whisper}"
                    logging.info(s)
            except asyncio.CancelledError:
                logging.warning(f"Cancelling {scope['request']['uuid']} due to disconnect")
            finally:

                if 'cache' in scope and scope['cache'] is not None:
                    cleanup_cache(scope['cache'])
                    scope.pop('cache', None)

                torch.cuda.empty_cache()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        scope['request'] = {
            'uuid': str(uuid.uuid4()),
            'before_queue': time.time()
        }

        if self.semaphore.locked():
            logging.debug(f"{scope['request']['uuid']} waiting for queue.")
            future = asyncio.Future()
            self.queue.append(future)
            await future

        scope['request']['after_queue'] = time.time()
        scope['request']['time_in_queue'] = scope['request']['after_queue'] - \
            scope['request']['before_queue']

        await self.process_request(scope, receive, send)

        if self.queue:
            next_request = self.queue.popleft()
            next_request.set_result(None)


app = FastAPI()

app.add_middleware(InsertMiddleware, max_concurrent=args.max_concurrent)

if args.serving_type == 'chat':
    @app.post('/chat/completions')
    async def chat_completions_main(
        form: ChatCompletionForm,
        request: Request = None,
    ):
        generator = chat_completions(form=form, request=request)
        r = await generator
        if form.stream:
            return EventSourceResponse(r, headers=HEADERS)
        else:
            return r

if args.serving_type == 'whisper':
    @app.post('/audio/transcriptions')
    async def audio_transcriptions_main(
        file: bytes = File(),
        model: str = Form('base'),
        language: str = Form(None),
        response_format: str = Form('text'),
        timestamp_granularities: str = Form('segment'),
        stream: bool = Form(False),
        repetition_penalty: float = Form(1.0),
        temperature: float = Form(0.0),
        top_p: float = Form(0.95),
        top_k: int = Form(50),
        request: Request = None,
    ):
        if isinstance(language, str):
            language = language.lower().strip()
            if language in {'null', 'none'}:
                language = None

        generator = audio_completions(
            file=file,
            language=language,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities,
            stream=stream,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            request=request,
        )
        r = await generator
        if stream:
            return EventSourceResponse(r, headers=HEADERS)
        else:
            return r

if args.continous_batching:
    @app.on_event("startup")
    async def startup_event():
        app.state.background_prefill = asyncio.create_task(prefill())
        app.state.background_step = asyncio.create_task(step())

    @app.on_event("shutdown")
    async def shutdown_event():
        app.state.background_prefill.cancel()
        app.state.background_step.cancel()
        try:
            await app.state.background_prefill
        except asyncio.CancelledError:
            pass
        try:
            await app.state.background_step
        except asyncio.CancelledError:
            pass

if args.hotload:
    load_model()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.loglevel.lower(),
        access_log=True,
    )

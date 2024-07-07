from transformers_openai.env import args
from transformers_openai.base_model import ChatCompletionForm
import asyncio
import logging
import uuid
import time
import torch
import uvicorn
from fastapi import FastAPI, Request
from sse_starlette import EventSourceResponse
from transformers import cache_utils
from collections import deque

if args.accelerator_type == 'cuda':
    from transformers_openai.main_cuda import chat_completions, load_model

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
                time_taken_first_token = scope['request']['time_first_token'] - \
                    scope['request']['after_queue']
                time_taken_max_tokens = scope['request']['time_max_tokens'] - \
                    scope['request']['time_first_token']
                tps = scope['request']['total_tokens'] / time_taken_max_tokens
                logging.info(
                    f"Complete {scope['request']['uuid']}, time first token {time_taken_first_token} seconds, time taken {time_taken_max_tokens} seconds, TPS {tps}")
            except asyncio.CancelledError:
                logging.warning(f"Cancelling {scope['request']['uuid']} due to disconnect")
            finally:

                if 'cache' in scope and scope['cache'] is not None:

                    if isinstance(scope['cache'], tuple):
                        scope['cache'] = list(scope['cache'])
                        for i in range(len(scope['cache'])):
                            scope['cache'][i] = list(scope['cache'][i])
                            for _ in range(len(scope['cache'][i])):
                                del scope['cache'][i][0]

                    else:
                        for _ in range(len(scope['cache'].key_cache)):
                            del scope['cache'].key_cache[0]
                        for _ in range(len(scope['cache'].value_cache)):
                            del scope['cache'].value_cache[0]

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

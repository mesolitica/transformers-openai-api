from env import MAX_CONCURRENT, ACCELERATOR_TYPE, HEADERS, HOTLOAD, CACHE_TYPE
from app.base_model import ChatCompletionForm
import asyncio
import logging
import uuid
import time
import torch
from fastapi import FastAPI, Request
from sse_starlette import EventSourceResponse
from transformers import cache_utils
from transformers.cache_utils import DynamicCache
from collections import deque

if ACCELERATOR_TYPE == 'cuda':
    from app.main_cuda import chat_completions, load_model


class InsertMiddleware:
    def __init__(self, app, max_concurrent=50):
        self.app = app
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue = deque()

    async def process_request(self, scope, receive, send):
        async with self.semaphore:

            scope['cache'] = getattr(cache_utils, CACHE_TYPE, None)
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
                return await handler_task
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
        logging.debug(
            f"{scope['request']['uuid']} stay in queue for {scope['request']['time_in_queue']} seconds")

        await self.process_request(scope, receive, send)

        if self.queue:
            next_request = self.queue.popleft()
            next_request.set_result(None)


app = FastAPI()

app.add_middleware(InsertMiddleware, max_concurrent=MAX_CONCURRENT)


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

if HOTLOAD:
    load_model()

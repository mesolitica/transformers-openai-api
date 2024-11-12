from transformers_openai.env import args
from transformers_openai.base_model import ChatCompletionForm
from transformers_openai.middleware import InsertMiddleware, HEADERS
from fastapi import FastAPI, Request
from fastapi import File, Form, UploadFile
from sse_starlette import EventSourceResponse
import asyncio
import uvicorn

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

@app.on_event("startup")
async def startup_event():
    app.state.background_prefill = asyncio.create_task(prefill())
    app.state.background_step = asyncio.create_task(step())
    load_model()

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

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.loglevel.lower(),
        access_log=True,
    )

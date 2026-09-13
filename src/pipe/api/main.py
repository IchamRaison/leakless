"""Application CPU locale. Les adaptateurs ML seront intégrés après livraison G1/G2."""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from pipe.api.audio import MAX_BYTES, MAX_SAMPLES, AudioError, decode_audio, visualization
from pipe.contracts import PredictRequest, Prediction, Sample


def fail(status: int, code: str, message: str):
    raise HTTPException(status, {"code": code, "message": message})


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.samples = {}
    yield
    app.state.samples.clear()


app = FastAPI(title="LeakLess · Acoustic API", version="0.1.0", lifespan=lifespan)


class BoundUpload:
    """Plafond avant parsing multipart, y compris les corps sans Content-Length."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        limit = MAX_BYTES + 64 * 1024
        content_length = dict(scope["headers"]).get(b"content-length")
        too_large = JSONResponse({"error": {"code": "payload_too_large", "message": "File limited to 8 MiB."}}, status_code=413)
        if content_length and (not content_length.isdigit() or int(content_length) > limit):
            await too_large(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > limit:
                await too_large(scope, receive, send)
                return
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)


app.add_middleware(BoundUpload)


@app.middleware("http")
async def no_cache(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"code": "http_error", "message": str(exc.detail)}
    return JSONResponse({"error": detail}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse({"error": {"code": "invalid_request", "message": "Invalid request. Check the required fields."}}, status_code=422)


def get_sample(sample_id: str):
    audio = app.state.samples.get(sample_id)
    if audio is None:
        fail(404, "sample_not_found", "Recording not found. Upload it again if the server restarted.")
    return audio


@app.get("/health")
def health():
    return {"status": "ok", "schema_version": "0.1", "device": "cpu",
            "models": [{"name": name, "available": False, "version": None,
                        "reason": "Adapter and weights not delivered"} for name in ("tslm", "baseline")],
            "capabilities": {"upload": True, "visualization": True, "perturbation": False, "replay": False}}


@app.get("/samples", response_model=list[Sample])
def samples():
    return [audio.metadata for audio in app.state.samples.values()]


@app.post("/samples", response_model=Sample, status_code=201)
async def upload(file: Annotated[UploadFile, File()]):
    try:
        if not file.filename or not file.filename.lower().endswith(".wav"):
            fail(415, "unsupported_audio", "Choose a .wav file.")
        raw = await file.read(MAX_BYTES + 1)
    finally:
        await file.close()
    if len(raw) > MAX_BYTES:
        fail(413, "payload_too_large", "File limited to 8 MiB.")
    try:
        audio = decode_audio(raw)
    except AudioError as exc:
        fail(422, "invalid_audio", str(exc))
    if audio.metadata.sample_id not in app.state.samples and len(app.state.samples) >= MAX_SAMPLES:
        fail(409, "sample_limit", "16 recordings maximum. Remove one before uploading another.")
    app.state.samples[audio.metadata.sample_id] = audio
    return audio.metadata


@app.delete("/samples/{sample_id}", status_code=204)
def remove_sample(sample_id: str):
    get_sample(sample_id)
    del app.state.samples[sample_id]
    return Response(status_code=204)


@app.get("/samples/{sample_id}/audio")
def audio_file(sample_id: str):
    audio = get_sample(sample_id)
    return Response(audio.original, media_type="audio/wav", headers={"X-Input-SHA256": audio.metadata.input_sha256})


@app.get("/samples/{sample_id}/visualization")
def visualize(sample_id: str):
    return visualization(get_sample(sample_id))


@app.get("/samples/{sample_id}/label")
def label(sample_id: str):
    get_sample(sample_id)
    fail(404, "label_unavailable", "No authorised demonstration label for this upload.")


@app.post("/predict", response_model=Prediction)
def predict(body: PredictRequest):
    get_sample(body.sample_id)
    fail(503, "model_unavailable", "Model unavailable: adapter and weights not delivered.")


@app.get("/evaluation")
def evaluation():
    fail(404, "evaluation_unavailable", "Not evaluated: no published run metrics.json is integrated.")

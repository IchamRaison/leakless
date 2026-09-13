"""Endpoint GPU explicite, un modèle chargé, aucune notification externe."""
from datetime import datetime
import hashlib
import hmac
import os
from pathlib import Path
import sqlite3
from threading import Lock
import time
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, Header, Request, UploadFile
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException
from starlette.concurrency import run_in_threadpool

from pipe.temporal_model import C1Detector, decode_pcm
from pipe.temporal_tracking import Tracker


def protect(authorization: Annotated[str | None, Header()] = None):
    token = os.getenv("PIPE_TEMPORAL_TOKEN")
    if token and not hmac.compare_digest(authorization or "", "Bearer " + token):
        raise HTTPException(401, {"code": "unauthorized", "message": "Authentication required."})


router = APIRouter(prefix="/temporal", dependencies=[Depends(protect)])


class SessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_mode: Literal["replay", "development_fixture"] = "replay"


def initialize(app):
    app.state.temporal = None
    app.state.temporal_reason = "PIPE_TEMPORAL_BUNDLE, SHA256 and DB are required."
    values = [os.getenv("PIPE_TEMPORAL_" + key) for key in ("BUNDLE", "SHA256", "DB")]
    if not all(values):
        return
    try:
        detector = C1Detector(values[0], values[1])
        tracker = Tracker(Path(values[2]), detector.metadata["config"]["policy"], detector.version)
        app.state.temporal = {"detector": detector, "tracker": tracker, "lock": Lock(), "sequence": None}
        app.state.temporal_reason = None
    except Exception as exc:
        app.state.temporal_reason = f"Temporal bundle unavailable ({type(exc).__name__})."


def service(request):
    value = request.app.state.temporal
    if value is None:
        raise HTTPException(503, {"code": "temporal_unavailable", "message": request.app.state.temporal_reason})
    return value


def checked(action):
    try:
        return action()
    except KeyError:
        raise HTTPException(404, {"code": "session_not_found", "message": "Unknown session."})
    except ValueError as exc:
        raise HTTPException(422, {"code": "invalid_temporal_input", "message": str(exc)})
    except sqlite3.Error:
        raise HTTPException(503, {"code": "storage_unavailable", "message": "Event storage unavailable; no delivery claimed."})


@router.get("/health")
def health(request: Request):
    ready = request.app.state.temporal
    return {"available": ready is not None, "reason": request.app.state.temporal_reason,
        "model_version": ready["detector"].version if ready else None,
        "schema_version": "pipe.temporal-api.v1", "notifications": "preview_only",
        "event_detector_validated": False, "sequence_model_status": "experimental_terminal_30s_only"}


@router.post("/sessions", status_code=201)
def create(request: Request, body: SessionRequest):
    return checked(lambda: service(request)["tracker"].create(body.source_mode))


@router.get("/sessions/{session_id}")
def status(request: Request, session_id: str):
    return checked(lambda: service(request)["tracker"].snapshot(session_id))


@router.post("/sessions/{session_id}/end")
def end(request: Request, session_id: str):
    return checked(lambda: service(request)["tracker"].end(session_id))


async def read_upload(file, limit):
    try:
        raw = await file.read(limit + 1)
    finally:
        await file.close()
    if len(raw) > limit:
        raise HTTPException(413, {"code": "audio_too_large", "message": "Audio payload exceeds endpoint limit."})
    return raw


@router.post("/sessions/{session_id}/windows")
async def window(request: Request, session_id: str, file: Annotated[UploadFile, File()],
                 sequence: Annotated[int, Form(ge=0)], source_end_at: Annotated[datetime, Form()]):
    value = service(request)
    checked(lambda: value["tracker"].snapshot(session_id))
    raw = await read_upload(file, 65536)
    if source_end_at.tzinfo is None:
        raise HTTPException(422, {"code": "timezone_required", "message": "Use an explicit UTC offset."})
    started = time.perf_counter()
    problem, features, score = None, None, None
    try:
        features, score = await run_in_threadpool(value["detector"].score, decode_pcm(raw))
        features = features.tolist()
    except ValueError as exc:
        problem = str(exc)
    result = checked(lambda: value["tracker"].consume(session_id, sequence, source_end_at.timestamp(),
        time.time(), hashlib.sha256(raw).hexdigest(), score, features, problem))
    return result | {"latency_ms": (time.perf_counter() - started) * 1000}


@router.post("/sequence")
async def sequence(request: Request, file: Annotated[UploadFile, File()]):
    value = service(request)
    raw = await read_upload(file, 1024 * 1024)
    if not value["lock"].acquire(blocking=False):
        raise HTTPException(409, {"code": "model_busy", "message": "Sequence inference in progress; retry later."})
    started = time.perf_counter()
    try:
        def infer():
            from pipe.sequence_model import load_sequence, sequence_probability
            inputs = checked(lambda: value["detector"].sequence_inputs(decode_pcm(raw, seconds=30)))
            if value["sequence"] is None:
                value["sequence"] = load_sequence(value["detector"], os.getenv("PIPE_TEMPORAL_DEVICE", "cuda"))
            model, mean, scale = value["sequence"]
            return sequence_probability(model, inputs, mean, scale)
        score = await run_in_threadpool(infer)
        return {"audio_sha256": hashlib.sha256(raw).hexdigest(), "probability_leak": score,
            "model_version": "lstm-" + value["detector"].metadata["files"]["lstm.pt"][:12],
            "c1_model_version": value["detector"].version,
            "calibration": "none", "scope": "sequence_classification_30_seconds",
            "event_onsets_estimated": False, "experimental": True,
            "latency_ms": (time.perf_counter() - started) * 1000}
    except (RuntimeError, ImportError, OSError):
        raise HTTPException(503, {"code": "sequence_unavailable", "message": "Sequence inference unavailable; no score fabricated."})
    finally:
        value["lock"].release()

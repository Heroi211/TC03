"""API FastAPI: recebe texto de laudo e retorna classificação de urgência."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.logging_setup import get_logger
from src.optimize import OnnxTextClassifier

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SKLEARN = ROOT / "models" / "triagem_sklearn.joblib"
DEFAULT_ONNX = ROOT / "models" / "triagem.onnx"

MODEL_BACKEND = os.getenv("MODEL_BACKEND", "onnx").strip().lower()

logger = get_logger("triagem.api")

REQUESTS = Counter(
    "triagem_requests_total",
    "Total de requisições HTTP",
    ["method", "endpoint", "status"],
)
ERRORS = Counter(
    "triagem_errors_total",
    "Total de respostas de erro (4xx/5xx)",
    ["endpoint", "status"],
)
LATENCY = Histogram(
    "triagem_request_duration_seconds",
    "Duração das requisições HTTP",
    ["endpoint"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

_model: Any | None = None
_backend: str = MODEL_BACKEND


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texto do laudo médico")


class PredictResponse(BaseModel):
    label: str
    latency_ms: float
    backend: str


def load_model(backend: str = MODEL_BACKEND) -> Any:
    if backend == "onnx":
        if not DEFAULT_ONNX.exists():
            raise FileNotFoundError(
                f"Modelo ONNX não encontrado em {DEFAULT_ONNX}. "
                "Execute: python -m src.optimize"
            )
        logger.info("Carregando modelo ONNX: %s", DEFAULT_ONNX)
        return OnnxTextClassifier(DEFAULT_ONNX)

    if backend == "sklearn":
        if not DEFAULT_SKLEARN.exists():
            raise FileNotFoundError(
                f"Modelo sklearn não encontrado em {DEFAULT_SKLEARN}. "
                "Execute: python -m src.train"
            )
        logger.info("Carregando modelo sklearn: %s", DEFAULT_SKLEARN)
        return joblib.load(DEFAULT_SKLEARN)

    raise ValueError(f"MODEL_BACKEND inválido: {backend!r}. Use 'sklearn' ou 'onnx'.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _model, _backend
    _backend = MODEL_BACKEND
    logger.info("Subindo API | backend=%s", _backend)
    _model = load_model(_backend)
    logger.info("Modelo carregado com sucesso")
    yield
    _model = None
    logger.info("API encerrada")


app = FastAPI(
    title="Triagem de Laudos",
    description="Classificador de urgência de exames de texto (NLP leve).",
    version="0.3.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    endpoint = request.url.path
    if endpoint == "/metrics":
        return await call_next(request)

    started = time.perf_counter()
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        elapsed = time.perf_counter() - started
        LATENCY.labels(endpoint=endpoint).observe(elapsed)
        REQUESTS.labels(
            method=request.method,
            endpoint=endpoint,
            status=status,
        ).inc()
        if status.startswith(("4", "5")):
            ERRORS.labels(endpoint=endpoint, status=status).inc()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "backend": _backend}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if _model is None:
        logger.error("Predição recusada: modelo não carregado")
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    started = time.perf_counter()
    label = str(_model.predict([payload.text])[0])
    latency_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "predict | label=%s | latency_ms=%.3f | backend=%s",
        label,
        latency_ms,
        _backend,
    )
    return PredictResponse(
        label=label,
        latency_ms=round(latency_ms, 3),
        backend=_backend,
    )

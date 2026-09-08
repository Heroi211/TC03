"""API FastAPI: recebe texto de laudo e retorna classificação de urgência."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.optimize import OnnxTextClassifier

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SKLEARN = ROOT / "models" / "triagem_sklearn.joblib"
DEFAULT_ONNX = ROOT / "models" / "triagem.onnx"

# sklearn (baseline) | onnx (otimizado)
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "onnx").strip().lower()

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
        return OnnxTextClassifier(DEFAULT_ONNX)

    if backend == "sklearn":
        if not DEFAULT_SKLEARN.exists():
            raise FileNotFoundError(
                f"Modelo sklearn não encontrado em {DEFAULT_SKLEARN}. "
                "Execute: python -m src.train"
            )
        return joblib.load(DEFAULT_SKLEARN)

    raise ValueError(f"MODEL_BACKEND inválido: {backend!r}. Use 'sklearn' ou 'onnx'.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _model, _backend
    _backend = MODEL_BACKEND
    _model = load_model(_backend)
    yield
    _model = None


app = FastAPI(
    title="Triagem de Laudos",
    description="Classificador de urgência de exames de texto (NLP leve).",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "backend": _backend}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    started = time.perf_counter()
    label = str(_model.predict([payload.text])[0])
    latency_ms = (time.perf_counter() - started) * 1000
    return PredictResponse(
        label=label,
        latency_ms=round(latency_ms, 3),
        backend=_backend,
    )

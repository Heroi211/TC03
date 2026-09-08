"""Testes básicos da API de triagem."""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.api import main as api_main


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    texts = [
        "exame normal sem alteracoes",
        "acompanhamento necessario atencao",
        "emergencia urgente risco imediato",
    ]
    labels = ["normal", "atencao", "urgente"]
    pipeline.fit(texts, labels)

    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)
    monkeypatch.setenv("MODEL_BACKEND", "sklearn")
    monkeypatch.setattr(api_main, "MODEL_BACKEND", "sklearn")
    monkeypatch.setattr(api_main, "DEFAULT_SKLEARN", model_path)
    api_main._model = None
    api_main._backend = "sklearn"

    with TestClient(api_main.app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["backend"] == "sklearn"


def test_predict_returns_label(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "exame normal sem alteracoes"})
    assert response.status_code == 200
    body = response.json()
    assert "label" in body
    assert body["label"] in {"normal", "atencao", "urgente"}
    assert body["latency_ms"] >= 0
    assert body["backend"] == "sklearn"

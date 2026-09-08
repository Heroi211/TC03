"""Converte o pipeline sklearn para ONNX e compara latência de inferência."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SKLEARN = ROOT / "models" / "triagem_sklearn.joblib"
DEFAULT_ONNX = ROOT / "models" / "triagem.onnx"

SAMPLES = [
    "exame dentro dos limites da normalidade sem alteracoes significativas",
    "alteracao leve que merece acompanhamento medico em curto prazo",
    "sinais de emergencia com risco imediato ao paciente",
]


class OnnxTextClassifier:
    """Wrapper mínimo para manter interface .predict([texto])."""

    def __init__(self, model_path: Path):
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        # label costuma ser a primeira saída; probabilidades a segunda (se existir)
        self.label_index = 0

    def predict(self, texts: list[str]) -> list[str]:
        # skl2onnx espera matriz de strings shape (n, 1)
        arr = np.array([[t] for t in texts], dtype=object)
        outputs = self.session.run(None, {self.input_name: arr})
        labels = outputs[self.label_index]
        return [str(x) for x in labels]


def export_onnx(
    sklearn_path: Path = DEFAULT_SKLEARN,
    onnx_path: Path = DEFAULT_ONNX,
) -> Path:
    if not sklearn_path.exists():
        raise FileNotFoundError(
            f"Modelo sklearn não encontrado em {sklearn_path}. "
            "Execute: python -m src.train"
        )

    pipeline = joblib.load(sklearn_path)
    initial_types = [("input", StringTensorType([None, 1]))]
    # Locale "C" evita dependência de en_US.UTF-8 (comum em WSL/containers mínimos).
    options = {
        TfidfVectorizer: {"locale": "C"},
        LogisticRegression: {"zipmap": False},
    }
    onx = convert_sklearn(
        pipeline,
        initial_types=initial_types,
        target_opset=12,
        options=options,
    )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(onx.SerializeToString())
    print(f"Modelo ONNX salvo em: {onnx_path}")
    return onnx_path


def _time_predict(predict_fn: Any, texts: list[str], rounds: int) -> list[float]:
    # warmup
    predict_fn(texts[:1])
    latencies: list[float] = []
    for i in range(rounds):
        sample = [texts[i % len(texts)]]
        started = time.perf_counter()
        predict_fn(sample)
        latencies.append((time.perf_counter() - started) * 1000)
    return latencies


def compare_latency(
    sklearn_path: Path = DEFAULT_SKLEARN,
    onnx_path: Path = DEFAULT_ONNX,
    rounds: int = 200,
) -> dict[str, float]:
    sklearn_model = joblib.load(sklearn_path)
    onnx_model = OnnxTextClassifier(onnx_path)

    sk_lat = _time_predict(sklearn_model.predict, SAMPLES, rounds)
    onnx_lat = _time_predict(onnx_model.predict, SAMPLES, rounds)

    sk_mean = statistics.mean(sk_lat)
    onnx_mean = statistics.mean(onnx_lat)
    speedup = sk_mean / onnx_mean if onnx_mean > 0 else float("inf")

    print(f"Rodadas: {rounds}")
    print(f"sklearn  — média: {sk_mean:.3f} ms | p50: {statistics.median(sk_lat):.3f} ms")
    print(f"ONNX     — média: {onnx_mean:.3f} ms | p50: {statistics.median(onnx_lat):.3f} ms")
    print(f"Speedup (sklearn/onnx): {speedup:.2f}x")

    # Sanity: mesma previsão no mesmo texto
    text = SAMPLES[0]
    sk_label = str(sklearn_model.predict([text])[0])
    onnx_label = onnx_model.predict([text])[0]
    print(f"Checagem de label ('{text[:40]}...'): sklearn={sk_label} | onnx={onnx_label}")

    return {
        "sklearn_mean_ms": round(sk_mean, 3),
        "sklearn_p50_ms": round(statistics.median(sk_lat), 3),
        "onnx_mean_ms": round(onnx_mean, 3),
        "onnx_p50_ms": round(statistics.median(onnx_lat), 3),
        "speedup": round(speedup, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta ONNX e compara latência.")
    parser.add_argument("--sklearn", type=Path, default=DEFAULT_SKLEARN)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Só compara, sem reconverter.",
    )
    args = parser.parse_args()

    if not args.skip_export:
        export_onnx(args.sklearn, args.onnx)
    elif not args.onnx.exists():
        raise FileNotFoundError(f"ONNX não encontrado: {args.onnx}")

    compare_latency(args.sklearn, args.onnx, rounds=args.rounds)


if __name__ == "__main__":
    main()

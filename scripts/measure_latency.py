"""Mede latência média da API local (baseline)."""

from __future__ import annotations

import argparse
import statistics
import time

import httpx

SAMPLES = [
    "exame dentro dos limites da normalidade sem alteracoes significativas",
    "alteracao leve que merece acompanhamento medico em curto prazo",
    "sinais de emergencia com risco imediato ao paciente",
]


def measure(base_url: str, n: int) -> None:
    url = f"{base_url.rstrip('/')}/predict"
    latencies: list[float] = []

    with httpx.Client(timeout=30.0) as client:
        for i in range(n):
            text = SAMPLES[i % len(SAMPLES)]
            started = time.perf_counter()
            response = client.post(url, json={"text": text})
            elapsed_ms = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            latencies.append(elapsed_ms)

    print(f"Requisições: {n}")
    print(f"Latência média (ms): {statistics.mean(latencies):.3f}")
    print(f"Latência p50 (ms): {statistics.median(latencies):.3f}")
    print(f"Latência min/max (ms): {min(latencies):.3f} / {max(latencies):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline de latência da API.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    measure(args.url, args.n)


if __name__ == "__main__":
    main()

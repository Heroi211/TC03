"""Gera dataset sintético, treina classificador TF-IDF + LogisticRegression e salva o modelo."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "laudos.csv"
DEFAULT_MODEL = ROOT / "models" / "triagem_sklearn.joblib"

LABELS = ("normal", "atencao", "urgente")

# Frases-base por classe (textos curtos no estilo de laudo/sintoma).
_TEMPLATES: dict[str, list[str]] = {
    "normal": [
        "exame dentro dos limites da normalidade sem alteracoes significativas",
        "laudo sem achados patologicos relevantes paciente estavel",
        "resultado laboratorial compativel com valores de referencia",
        "imagem sem sinais de inflamacao ou lesao aguda",
        "avaliacao clinica rotineira sem queixas urgentes",
        "parametros vitais estaveis e exame fisico sem particularidades",
        "controle periodico sem indicacao de intervencao imediata",
        "achados benignos sem necessidade de conduta emergencial",
    ],
    "atencao": [
        "alteracao leve que merece acompanhamento medico em curto prazo",
        "resultado limítrofe sugerindo reavaliacao clinica em dias",
        "sinais moderados de inflamacao sem criterio de emergencia",
        "exame com achados inespecificos requer correlacao clinica",
        "sintomas persistentes com necessidade de investigacao adicional",
        "valores laboratoriais elevados de forma leve a moderada",
        "quadro subagudo com orientacao de retorno ambulatorial breve",
        "achado duvidoso recomendando exames complementares",
    ],
    "urgente": [
        "sinais de emergencia com risco imediato ao paciente",
        "alteracao critica exigindo avaliacao medica imediata",
        "quadro agudo grave com potencial de deterioracao rapida",
        "marcadores extremamente elevados sugestivos de urgencia",
        "suspeita de condicao ameacadora a vida conduta imediata",
        "instabilidade hemodinamica e sinais de choque clinico",
        "hemorragia ativa com necessidade de intervencao urgente",
        "falencia organica iminente priorizar atendimento emergencial",
    ],
}


def generate_dataset(n_samples: int = 2400, seed: int = 42) -> pd.DataFrame:
    """Gera CSV tabular texto + target com pelo menos ~2000 amostras."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows: list[dict[str, str]] = []
    per_class = n_samples // len(LABELS)

    for label in LABELS:
        templates = _TEMPLATES[label]
        for i in range(per_class):
            base = templates[i % len(templates)]
            noise = rng.integers(0, 1000)
            text = f"{base} codigo_{noise} amostra_{i}"
            rows.append({"text": text, "target": label})

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )


def prepare_dataframe(
    data_path: Path = DEFAULT_DATA,
    n_samples: int = 2400,
) -> pd.DataFrame:
    """Carrega o CSV de treino ou gera o dataset sintético se ainda não existir."""
    data_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        df = generate_dataset(n_samples=n_samples)
        df.to_csv(data_path, index=False)
        print(f"Dataset gerado: {data_path} ({len(df)} linhas)")
    else:
        df = pd.read_csv(data_path)
        print(f"Dataset carregado: {data_path} ({len(df)} linhas)")

    if "text" not in df.columns or "target" not in df.columns:
        raise ValueError("O CSV deve conter as colunas 'text' e 'target'.")
    return df


def fit_and_persist(
    df: pd.DataFrame,
    model_path: Path = DEFAULT_MODEL,
) -> Path:
    """Treina o pipeline e salva o artefato em model_path."""
    model_path.parent.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["target"],
        test_size=0.2,
        random_state=42,
        stratify=df["target"],
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(pipeline, model_path)
    print(f"Modelo salvo em: {model_path}")
    return model_path


def train(
    data_path: Path = DEFAULT_DATA,
    model_path: Path = DEFAULT_MODEL,
    n_samples: int = 2400,
) -> Path:
    df = prepare_dataframe(data_path=data_path, n_samples=n_samples)
    return fit_and_persist(df, model_path=model_path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o classificador de triagem.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--n-samples", type=int, default=2400)
    args = parser.parse_args()
    train(data_path=args.data, model_path=args.model, n_samples=args.n_samples)


if __name__ == "__main__":
    main()

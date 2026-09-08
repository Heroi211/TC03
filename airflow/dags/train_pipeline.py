"""
DAG Airflow — pipeline simples de treino/retreino.

Etapas (como no Tech Challenge):
  1. carregar_dados  → lê/gera CSV (text + target)
  2. treinar_modelo  → treina TF-IDF + Logistic Regression
  3. salvar_modelo   → persiste o artefato final em models/
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "laudos.csv"
STAGING_MODEL = PROJECT_ROOT / "models" / "triagem_sklearn.staging.joblib"
FINAL_MODEL = PROJECT_ROOT / "models" / "triagem_sklearn.joblib"


def carregar_dados(**context) -> str:
    from src.train import prepare_dataframe

    df = prepare_dataframe(data_path=DATA_PATH)
    context["ti"].xcom_push(key="n_rows", value=int(len(df)))
    context["ti"].xcom_push(key="data_path", value=str(DATA_PATH))
    print(f"[carregar_dados] {len(df)} linhas em {DATA_PATH}")
    return str(DATA_PATH)


def treinar_modelo(**context) -> str:
    import pandas as pd
    from src.train import fit_and_persist

    data_path = Path(context["ti"].xcom_pull(task_ids="carregar_dados", key="data_path"))
    df = pd.read_csv(data_path)
    fit_and_persist(df, model_path=STAGING_MODEL)
    context["ti"].xcom_push(key="staging_model", value=str(STAGING_MODEL))
    print(f"[treinar_modelo] artefato intermediário: {STAGING_MODEL}")
    return str(STAGING_MODEL)


def salvar_modelo(**context) -> str:
    staging = Path(
        context["ti"].xcom_pull(task_ids="treinar_modelo", key="staging_model")
    )
    if not staging.exists():
        raise FileNotFoundError(f"Modelo intermediário não encontrado: {staging}")

    FINAL_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(staging, FINAL_MODEL)
    staging.unlink(missing_ok=True)
    print(f"[salvar_modelo] modelo final: {FINAL_MODEL}")
    return str(FINAL_MODEL)


default_args = {
    "owner": "tech-challenge",
    "depends_on_past": False,
}

with DAG(
    dag_id="triagem_treino_pipeline",
    description="Carrega CSV, treina classificador de triagem e salva o modelo.",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,  # disparo manual / sob demanda
    catchup=False,
    tags=["triagem", "treino", "tech-challenge"],
) as dag:
    t_carregar = PythonOperator(
        task_id="carregar_dados",
        python_callable=carregar_dados,
    )
    t_treinar = PythonOperator(
        task_id="treinar_modelo",
        python_callable=treinar_modelo,
    )
    t_salvar = PythonOperator(
        task_id="salvar_modelo",
        python_callable=salvar_modelo,
    )

    t_carregar >> t_treinar >> t_salvar

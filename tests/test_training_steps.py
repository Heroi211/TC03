"""Valida a lógica das etapas da DAG sem subir o scheduler do Airflow."""

from __future__ import annotations

from pathlib import Path

from src.train import fit_and_persist, prepare_dataframe


def test_training_steps_load_train_save(tmp_path: Path) -> None:
    data_path = tmp_path / "laudos.csv"
    staging = tmp_path / "model.staging.joblib"
    final = tmp_path / "model.joblib"

    df = prepare_dataframe(data_path=data_path, n_samples=60)
    assert data_path.exists()
    assert len(df) == 60
    assert set(df["target"]) == {"normal", "atencao", "urgente"}

    fit_and_persist(df, model_path=staging)
    assert staging.exists()

    final.write_bytes(staging.read_bytes())
    assert final.exists()
    assert final.stat().st_size > 0

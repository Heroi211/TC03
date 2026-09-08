# Tech Challenge — Fase 3

Triagem automática de laudos médicos (texto) com classificador NLP leve, API REST em Docker, foco em ciclo de vida do modelo (treino, inferência, latência e monitoramento).

## Objetivo

Classificar urgência de exames de texto (`normal` / `atencao` / `urgente`) via API FastAPI empacotada em container Docker, com otimização de latência (ONNX), CI/CD, Airflow e monitoramento (Prometheus + Grafana).

## Decisão arquitetural de deploy em nuvem

**Premissa necessária para implementação:** o PDF exige análise textual (README), não o provisionamento real em cloud.

### Batch vs real-time

| Abordagem | Adequação ao cenário |
| --- | --- |
| **Real-time (API síncrona)** | Adequada à triagem na porta de entrada: o profissional envia o texto do laudo e recebe a classificação imediatamente. |
| **Batch** | Útil para reprocessar históricos ou filas noturnas de laudos acumulados; não atende bem a urgência clínica pontual. |

**Escolha para este cenário:** inferência **real-time** (API REST), com treino/retreino em pipeline separado (batch/orquestrado).

### Provedor (exemplo)

Qualquer um dos três atende; a escolha abaixo é didática:

- **AWS** como referência: serviço de containers gerenciado (ex.: ECS/Fargate ou App Runner) para a API; armazenamento do artefato do modelo em object storage (S3); treino periódico via orquestração (equivalente ao Airflow) em ambiente de jobs.
- Alternativas equivalentes: **Azure** (Container Apps + Blob) ou **GCP** (Cloud Run + Cloud Storage).

Justificativa resumida: o workload é I/O + CPU leve (TF-IDF + modelo clássico), com necessidade de baixa latência de resposta e escala horizontal simples — perfil típico de **API em container**, não de cluster pesado de GPU.

## Estrutura do projeto

```text
TC_03/
├── .github/workflows/ci.yml
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/                 # datasource + dashboard (≥3 painéis)
├── airflow/dags/train_pipeline.py
├── src/
│   ├── api/main.py              # FastAPI + métricas Prometheus
│   ├── logging_setup.py         # logs no stdout
│   ├── train.py
│   └── optimize.py
├── docker-compose.yml           # API + Prometheus + Grafana
├── Dockerfile
└── README.md
```

## Pré-requisitos

- Python 3.11+ (local)
- Docker + Docker Compose
- Conta/repositório GitHub (para o Actions)

## Como executar

### 1. Ambiente e dependências

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Treinar o modelo

```bash
python -m src.train
```

Gera `data/laudos.csv` (≥ 2000 amostras) e `models/triagem_sklearn.joblib`.

**Premissa:** dataset sintético texto + target (pode trocar por CSV real com colunas `text` e `target`).

### 3. Otimizar para ONNX

```bash
python -m src.optimize
```

### 4. API local

```bash
MODEL_BACKEND=onnx uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"sinais de emergencia com risco imediato ao paciente"}'
```

### 5. Testes e lint

```bash
ruff check src tests scripts
pytest -q
```

### 6. CI/CD (GitHub Actions)

`.github/workflows/ci.yml` — no push/PR: **lint (ruff)** + **pytest**.

### 7. Stack completa (API + Prometheus + Grafana)

Gere os modelos antes do build:

```bash
python -m src.train
python -m src.optimize
docker compose up --build
```

| Serviço | URL |
| --- | --- |
| API | http://localhost:8000 |
| Métricas | http://localhost:8000/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin; anônimo em modo Viewer) |

Dashboard provisionado: **Triagem API** com 3 painéis:

1. Total de requisições (req/s)
2. Latência (p50 / p95)
3. Taxa de erro (4xx/5xx)

JSON do dashboard: `monitoring/grafana/dashboards/triagem.json`

Gere tráfego para popular os gráficos:

```bash
python scripts/measure_latency.py --url http://127.0.0.1:8000 --n 50
```

Logs da API no terminal:

```bash
docker compose logs -f api
```

### 8. DAG Airflow

Arquivo: `airflow/dags/train_pipeline.py` — `carregar_dados → treinar_modelo → salvar_modelo`.

Ver seção anterior no histórico do README / `requirements-airflow.txt` para subir o Airflow em venv separado.

## Latência

### Baseline HTTP (API)

| Ambiente | Latência média | p50 |
| --- | --- | --- |
| Local (uvicorn) | 2.65 ms | 2.27 ms |
| Docker | 3.72 ms | 2.24 ms |

### Inferência: sklearn vs ONNX (200 rodadas)

| Backend | Média | p50 |
| --- | --- | --- |
| sklearn | 0.733 ms | 0.590 ms |
| ONNX | 0.031 ms | 0.029 ms |
| Speedup | **~23.8x** | — |

## Endpoint

| Método | Rota | Descrição |
| --- | --- | --- |
| GET | `/health` | Saúde + backend |
| GET | `/metrics` | Métricas Prometheus |
| POST | `/predict` | Classificação do laudo |

## Status

- [x] API FastAPI + Docker
- [x] Modelo sklearn + otimização ONNX
- [x] CI/CD (lint + test)
- [x] DAG Airflow
- [x] Logging stdout (terminal / docker logs)
- [x] Prometheus + Grafana (Compose, ≥ 3 painéis)
- [x] README / decisão cloud
- [ ] Vídeo STAR (≤ 5 min)

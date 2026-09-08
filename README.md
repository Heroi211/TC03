# Tech Challenge — Fase 3

Triagem automática de laudos médicos (texto) com classificador NLP leve, API REST em Docker, foco em ciclo de vida do modelo (treino, inferência, latência).

## Objetivo

Classificar urgência de exames de texto (`normal` / `atencao` / `urgente`) via API FastAPI empacotada em container Docker, com otimização de latência (ONNX), CI/CD e baseline documentado.

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

> Pendente: stack de monitoramento (Prometheus + Grafana via Docker Compose).

## Estrutura do projeto

```text
TC_03/
├── .github/workflows/ci.yml    # lint + pytest
├── data/                       # CSV texto + target (gerado no treino)
├── models/                     # Artefatos .joblib e .onnx
├── airflow/
│   └── dags/
│       └── train_pipeline.py   # DAG: carregar → treinar → salvar
├── src/
│   ├── api/main.py             # FastAPI de inferência
│   ├── train.py                # Geração de dados + treino
│   └── optimize.py             # Export ONNX + comparação de latência
├── scripts/
│   └── measure_latency.py
├── tests/
├── Dockerfile
├── requirements.txt
├── requirements-airflow.txt
└── README.md
```

## Pré-requisitos

- Python 3.11+ (local)
- Docker (para empacotar a API)
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

Gera `data/laudos.csv` (dataset sintético tabular, ≥ 2000 amostras) e `models/triagem_sklearn.joblib`.

**Premissa:** dataset sintético no estilo laudo/sintoma + target de urgência, para cumprir o requisito de classificação sem depender de credenciais de Kaggle. Pode ser trocado por CSV real com colunas `text` e `target`.

### 3. Otimizar para ONNX

```bash
python -m src.optimize
```

Gera `models/triagem.onnx` e imprime a comparação de latência sklearn vs ONNX Runtime.

### 4. Subir a API (local)

Por padrão a API usa o backend **onnx** (`MODEL_BACKEND=onnx`). Para o baseline sklearn:

```bash
MODEL_BACKEND=sklearn uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# ou
MODEL_BACKEND=onnx uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Exemplo de chamada:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"sinais de emergencia com risco imediato ao paciente"}'
```

Resposta esperada (exemplo):

```json
{"label":"urgente","latency_ms":0.05,"backend":"onnx"}
```

### 5. Testes e lint (local)

```bash
ruff check src tests scripts
pytest -q
```

### 6. CI/CD (GitHub Actions)

Arquivo: `.github/workflows/ci.yml`

No **push** e em **pull_request**, o workflow executa **2 automações**:

1. **Lint** (`ruff check`)
2. **Testes** (`pytest`)

Não é necessário configurar secrets para esse pipeline básico.

### 7. API em Docker

```bash
python -m src.train
python -m src.optimize
docker build -t triagem-api .
docker run --rm -p 8000:8000 -e MODEL_BACKEND=onnx triagem-api
```

Em outro terminal:

```bash
python scripts/measure_latency.py --url http://127.0.0.1:8000 --n 50
```

### 8. DAG Airflow (treino/retreino)

Arquivo: `airflow/dags/train_pipeline.py`

Fluxo: **carregar_dados → treinar_modelo → salvar_modelo**

**Premissa necessária para implementação:** Airflow fica fora da imagem da API (dependência pesada). Use um venv separado ou `AIRFLOW_HOME` apontando para este projeto.

```bash
python -m venv .venv-airflow
source .venv-airflow/bin/activate
pip install -r requirements.txt -r requirements-airflow.txt

export AIRFLOW_HOME="$(pwd)/airflow"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/airflow/dags"

airflow db migrate
airflow standalone
```

No UI (http://localhost:8080), habilite e dispare a DAG `triagem_treino_pipeline`.

As funções de cada task também podem ser exercitadas via `pytest` (`tests/test_training_steps.py`) sem subir o scheduler.

## Latência

### Baseline HTTP (API)

Medição com `scripts/measure_latency.py` (50 requisições, backend sklearn na época da Fase 1):

| Ambiente | Métrica | Valor |
| --- | --- | --- |
| Local (uvicorn) | Latência média | **2.65 ms** |
| Local (uvicorn) | Latência p50 | **2.27 ms** |
| Docker (`triagem-api`) | Latência média | **3.72 ms** |
| Docker (`triagem-api`) | Latência p50 | **2.24 ms** |

### Comparação de inferência: sklearn vs ONNX

Medição com `python -m src.optimize` (200 predições locais, sem overhead HTTP):

| Backend | Latência média | Latência p50 |
| --- | --- | --- |
| sklearn (original) | **0.733 ms** | **0.590 ms** |
| ONNX Runtime | **0.031 ms** | **0.029 ms** |
| Speedup | **~23.8x** | — |

Técnica aplicada: conversão do pipeline TF-IDF + Logistic Regression para ONNX (`skl2onnx`) e inferência via ONNX Runtime.

## Endpoint

| Método | Rota | Descrição |
| --- | --- | --- |
| GET | `/health` | Saúde do serviço + backend ativo |
| POST | `/predict` | Corpo `{"text":"..."}` → `label` + `latency_ms` + `backend` |

## Status

- [x] API FastAPI funcional
- [x] Modelo sklearn (TF-IDF + Logistic Regression)
- [x] Dockerfile
- [x] Baseline de latência (HTTP)
- [x] Otimização ONNX + comparação original vs otimizado
- [x] Decisão arquitetural (cloud) neste README
- [x] DAG Airflow (carregar → treinar → salvar)
- [x] CI/CD (GitHub Actions: lint + test)
- [ ] Prometheus + Grafana
- [ ] Vídeo STAR (≤ 5 min)

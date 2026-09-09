# Tech Challenge — um comando sobe tudo
# Uso: make rise   |   make rise up

VENV      := .venv
PYTHON    := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip
COMPOSE   ?= docker compose

.PHONY: rise up down logs status bootstrap models help

help:
	@echo "Comandos:"
	@echo "  make rise      Cria venv, deps, modelos e sobe a stack completa"
	@echo "  make rise up   Idem"
	@echo "  make down      Para os containers"
	@echo "  make logs      Logs da API"
	@echo "  make status    Status dos containers"

# `make rise up` → rise faz o trabalho; up é no-op.
rise: bootstrap models
	@echo "AIRFLOW_UID=$$(id -u)" > .env
	$(COMPOSE) up --build -d
	@echo ""
	@echo "Stack no ar:"
	@echo "  API         http://localhost:8000"
	@echo "  Métricas    http://localhost:8000/metrics"
	@echo "  Prometheus  http://localhost:9090"
	@echo "  Grafana     http://localhost:3000  (admin/admin)"
	@echo "  Airflow     http://localhost:8080  (admin/admin)"
	@echo "              DAG: triagem_treino_pipeline"
	@echo ""
	@echo "Venv: source $(VENV)/bin/activate"
	@echo "Logs API: make logs"
	@echo "Obs: o 1º start do Airflow pode levar 1–2 min (instala sklearn no container)."

up:
	@true

bootstrap:
	@if [ ! -x "$(PYTHON)" ]; then \
		echo ">> Criando venv em $(VENV)/ ..."; \
		python3 -m venv $(VENV); \
	else \
		echo ">> Venv já existe: $(VENV)/"; \
	fi
	@echo ">> Instalando/atualizando requirements.txt ..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo ">> Dependências OK (interpretador padrão do Make: $(PYTHON))"

models: bootstrap
	@if [ ! -f models/triagem_sklearn.joblib ] || [ ! -f models/triagem.onnx ]; then \
		echo ">> Gerando modelos (train + ONNX) ..."; \
		$(PYTHON) -m src.train; \
		$(PYTHON) -m src.optimize; \
	else \
		echo ">> Modelos já existem — pulando treino."; \
	fi

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f api

status:
	$(COMPOSE) ps

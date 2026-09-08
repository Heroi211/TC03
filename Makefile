# Tech Challenge — atalhos de baixo esforço
# Uso: make rise up   (ou apenas: make rise)

PYTHON ?= python
COMPOSE ?= docker compose

.PHONY: rise up down logs status models help

help:
	@echo "Comandos:"
	@echo "  make rise up   Sobe a stack completa (API + Prometheus + Grafana)"
	@echo "  make rise      Idem"
	@echo "  make down      Para e remove os containers"
	@echo "  make logs      Acompanha logs da API"
	@echo "  make status    Mostra status dos containers"
	@echo "  make models    Só gera treino + ONNX (sem Compose)"

# `make rise up` passa dois alvos: o trabalho fica em `rise`; `up` é no-op.
rise: models
	$(COMPOSE) up --build -d
	@echo ""
	@echo "Stack no ar:"
	@echo "  API         http://localhost:8000"
	@echo "  Métricas    http://localhost:8000/metrics"
	@echo "  Prometheus  http://localhost:9090"
	@echo "  Grafana     http://localhost:3000  (admin/admin)"
	@echo ""
	@echo "Logs: make logs"

up:
	@true

models:
	@if [ ! -f models/triagem_sklearn.joblib ] || [ ! -f models/triagem.onnx ]; then \
		echo ">> Gerando modelos (train + ONNX)..."; \
		if [ -x .venv/bin/python ]; then \
			.venv/bin/python -m src.train; \
			.venv/bin/python -m src.optimize; \
		elif command -v $(PYTHON) >/dev/null 2>&1; then \
			$(PYTHON) -m src.train; \
			$(PYTHON) -m src.optimize; \
		else \
			echo "Python não encontrado. Crie o venv: python -m venv .venv && .venv/bin/pip install -r requirements.txt"; \
			exit 1; \
		fi; \
	else \
		echo ">> Modelos já existem — pulando treino."; \
	fi

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f api

status:
	$(COMPOSE) ps

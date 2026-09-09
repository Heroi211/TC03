# Tech Challenge — um comando sobe tudo
# Uso: make rise | make rise up | make fresh

VENV      := .venv
PYTHON    := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip
COMPOSE   ?= docker compose

.PHONY: rise up down clean fresh logs status bootstrap models help

help:
	@echo "Comandos:"
	@echo "  make rise      Sobe a stack (reusa cache/venv/modelos) — uso diário"
	@echo "  make rise up   Idem"
	@echo "  make fresh     Limpa Docker do projeto e sobe do zero"
	@echo "  make clean     Para containers, remove volumes órfãos e imagem da API"
	@echo "  make down      Só para os containers (mantém imagens/cache)"
	@echo "  make logs      Logs da API"
	@echo "  make status    Status dos containers"

# --- Uso diário: NÃO zera cache ---
# `make rise up` → rise faz o trabalho; up é no-op.
rise: bootstrap models
	@echo "AIRFLOW_UID=$$(id -u)" > .env
	$(COMPOSE) up --build -d
	@$(MAKE) --no-print-directory _urls

up:
	@true

# --- Instalação limpa sob demanda ---
# `make fresh` = clean + rise
# `make clean rise` também funciona (dois alvos)
fresh: clean rise

clean:
	@echo ">> Limpando stack Docker deste projeto..."
	-$(COMPOSE) down --remove-orphans --volumes
	-docker image rm -f tc_03-api 2>/dev/null || true
	@echo ">> Clean OK (venv e models/ no host são mantidos)."
	@echo "   Para zerar modelos também: rm -f models/*.joblib models/*.onnx data/*.csv"

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

_urls:
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

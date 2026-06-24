.PHONY: install lint format test train run-api help

help:
	@echo "Comandos disponíveis:"
	@echo "  install    Instala dependências com poetry"
	@echo "  lint       Verifica linting com ruff"
	@echo "  format     Formata código com ruff"
	@echo "  test       Executa testes com pytest"
	@echo "  train      Treina modelos e registra no MLflow"
	@echo "  run-api    Inicia a API FastAPI (porta 8000)"

install:
	poetry install

lint:
	poetry run ruff check src/ tests/
	poetry run ruff format --check src/ tests/

format:
	poetry run ruff format src/ tests/

test:
	poetry run pytest tests/ -v --tb=short

train:
	poetry run python -m customer_churn_ibm.train

run-api:
	poetry run uvicorn customer_churn_ibm.api.main:app --reload --host 0.0.0.0 --port 8000

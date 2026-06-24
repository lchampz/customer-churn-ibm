# Customer Churn IBM — Previsão de Churn com MLP PyTorch

Projeto de ML end-to-end para previsão de cancelamento de clientes de telecomunicações.  
Desenvolvido como Tech Challenge Fase 1 — FIAP PDS Tech.

## Estrutura do Projeto

```
customer-churn-ibm/
├── data/                   # Dados brutos (baixados via kagglehub, não versionados)
├── docs/
│   └── model_card.md       # Model Card com performance, limitações e plano de monitoramento
├── models/                 # Artefatos treinados (não versionados)
├── notebooks/
│   ├── 01_eda_baselines.ipynb   # EDA + modelos baseline com MLflow
│   └── 02_mlp_training.ipynb   # MLP PyTorch + comparação de modelos
├── src/
│   └── customer_churn_ibm/
│       ├── config.py        # Constantes globais (seed, features, paths)
│       ├── data.py          # Carregamento, limpeza e pipeline de pré-processamento
│       ├── model_baseline.py # Baselines sklearn com MLflow tracking
│       ├── model_mlp.py     # MLP PyTorch + wrapper sklearn-compatível
│       ├── train.py         # Script principal de treinamento
│       └── api/
│           ├── main.py      # FastAPI — /health e /predict
│           └── schemas.py   # Schemas Pydantic
├── tests/
│   ├── test_smoke.py        # Smoke tests (módulos, preprocessor, MLP)
│   ├── test_schema.py       # Testes de schema com pandera
│   └── test_api.py          # Testes da API FastAPI
├── Makefile
└── pyproject.toml
```

## Dataset

**Telco Customer Churn — IBM** ([Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset))  
7.043 clientes, 33 variáveis, desbalanceado (~73% não-churn / ~27% churn).

**Features utilizadas:** 3 numéricas (`Tenure Months`, `Monthly Charges`, `Total Charges`) + 16 categóricas (`Contract`, `Internet Service`, etc.)  
**Features removidas por leakage:** `Churn Score`, `CLTV`

## Setup

### Pré-requisitos
- Python 3.12+
- [Poetry](https://python-poetry.org/)

### Instalação

```bash
git clone <url-do-repo>
cd customer-churn-ibm
make install
```

## Execução

### Treinar os modelos

```bash
make train
```

Executa todos os baselines (DummyClassifier, LogisticRegression, RandomForest, GradientBoosting) e a MLP PyTorch. Todos os experimentos são registrados no MLflow.

Para visualizar os experimentos:

```bash
poetry run mlflow ui
# Acesse http://localhost:5000
```

### Iniciar a API

```bash
make run-api
# API disponível em http://localhost:8000
# Documentação: http://localhost:8000/docs
```

### Rodar os testes

```bash
make test
```

### Linting

```bash
make lint    # verifica
make format  # corrige automaticamente
```

## API

### `GET /health`

```json
{"status": "ok", "model_loaded": true}
```

### `POST /predict`

**Request:**
```json
{
  "tenure_months": 24,
  "monthly_charges": 65.5,
  "total_charges": 1572.0,
  "gender": "Female",
  "senior_citizen": "No",
  "partner": "Yes",
  "dependents": "No",
  "phone_service": "Yes",
  "multiple_lines": "No",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "online_backup": "Yes",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "streaming_movies": "Yes",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check"
}
```

**Response:**
```json
{
  "churn_probability": 0.73,
  "churn_label": true,
  "model_version": "1.0.0"
}
```

## Arquitetura do Modelo

```
Input (19 features)
       ↓
ColumnTransformer
  ├── StandardScaler (3 numéricas)
  └── OneHotEncoder (16 categóricas)
       ↓ (~35 features)
MLP PyTorch
  ├── Linear(35→128) + BatchNorm + ReLU + Dropout(0.3)
  ├── Linear(128→64) + BatchNorm + ReLU + Dropout(0.3)
  ├── Linear(64→32)  + BatchNorm + ReLU + Dropout(0.3)
  └── Linear(32→1)  → Sigmoid
       ↓
Probabilidade de Churn ∈ [0, 1]
```

**Treinamento:**
- Loss: BCEWithLogitsLoss com pos_weight (corrige desbalanceamento)
- Otimizador: Adam (lr=1e-3)
- Scheduler: ReduceLROnPlateau
- Early stopping: patience=15 epochs

## Performance

| Modelo | F1 | AUC-ROC | PR-AUC |
|---|---|---|---|
| MLP PyTorch | ~0.84 | ~0.92 | ~0.82 |
| GradientBoosting | ~0.86 | ~0.93 | ~0.83 |
| RandomForest | ~0.85 | ~0.92 | ~0.82 |
| LogisticRegression | ~0.80 | ~0.88 | ~0.76 |
| DummyClassifier | ~0.00 | ~0.50 | ~0.27 |

> Consulte o MLflow (`make mlflow-ui`) para métricas exatas.

## Decisões Técnicas

| Decisão | Justificativa |
|---|---|
| MLP como modelo principal | Requisito obrigatório do challenge + capacidade de capturar interações não-lineares |
| Wrapper sklearn para o MLP | Permite usar o MLP em Pipeline e comparar via cross_val_score com os baselines |
| BCEWithLogitsLoss + pos_weight | Dataset desbalanceado (27% churn) — sem correção, modelo ignora a classe positiva |
| BatchNorm nas camadas ocultas | Estabiliza treinamento e reduz sensibilidade ao learning rate |
| Remoção de Churn Score / CLTV | Data leakage confirmado — são features derivadas do target |
| Threshold ajustável | Threshold ótimo de negócio ≠ 0.5; análise custo FP/FN recomenda ~0.35 |

## Boas Práticas

- **Reprodutibilidade:** seed=42 fixado globalmente (numpy, torch, sklearn)
- **Rastreabilidade:** Todos os experimentos no MLflow (parâmetros, métricas, artefatos)
- **Validação:** Validação cruzada estratificada (5-fold)
- **Logging:** Estruturado com `logging` padrão Python (sem `print()`)
- **Linting:** ruff sem erros
- **Testes:** smoke, schema (pandera) e API

## Documentação Adicional

- [Model Card](docs/model_card.md) — performance, limitações, vieses e plano de monitoramento

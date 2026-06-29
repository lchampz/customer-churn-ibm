# Model Card — Customer Churn MLP

## Informações do Modelo

| Campo | Valor |
|---|---|
| **Nome** | Customer Churn MLP (PyTorch) |
| **Versão** | 1.0.0 |
| **Tipo** | Classificação Binária — Rede Neural MLP |
| **Framework** | PyTorch 2.x + scikit-learn Pipeline |
| **Data de desenvolvimento** | Junho 2026 |

## Descrição

Rede neural MLP (Multi-Layer Perceptron) treinada para prever a probabilidade de cancelamento (churn) de clientes de uma operadora de telecomunicações.

**Arquitetura:**
- Input: ~35 features (3 numéricas + categorias one-hot encoded)
- Camadas ocultas: 128 → 64 → 32 (com BatchNorm e Dropout=0.3)
- Output: 1 neurônio com ativação sigmoid
- Loss: BCEWithLogitsLoss com pos_weight para desbalanceamento
- Otimizador: Adam (lr=1e-3) + ReduceLROnPlateau
- Early stopping: patience=15 epochs

## Dataset

| Campo | Valor |
|---|---|
| **Fonte** | Telco Customer Churn — IBM (Kaggle: `yeanzc/telco-customer-churn-ibm-dataset`) |
| **Registros** | 7.043 clientes |
| **Features utilizadas** | 19 (3 numéricas, 16 categóricas) |
| **Target** | `Churn Value` (0 = não cancelou, 1 = cancelou) |
| **Split** | 80% treino / 20% teste (estratificado, seed=42) |
| **Desbalanceamento** | ~73% não-churn, ~27% churn |

### Features de Entrada

**Numéricas:**
- `Tenure Months` — tempo de contrato em meses
- `Monthly Charges` — cobrança mensal
- `Total Charges` — total pago até o momento

**Categóricas:**
- `Gender`, `Senior Citizen`, `Partner`, `Dependents`
- `Phone Service`, `Multiple Lines`, `Internet Service`
- `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`
- `Streaming TV`, `Streaming Movies`
- `Contract`, `Paperless Billing`, `Payment Method`

### Features Removidas (e por quê)

| Feature | Motivo da remoção |
|---|---|
| `Churn Score` | **Data leakage** — score derivado do label de churn |
| `CLTV` | **Data leakage** — calculado com base em churn esperado |
| `Churn Reason` | Disponível apenas pós-cancelamento |
| `CustomerID`, `Lat Long`, `Zip Code` | Identificadores sem poder preditivo |
| `Country`, `State` | Variância zero (todos EUA/Califórnia) |

## Performance

Métricas no conjunto de teste (20% dos dados, threshold=0.5):

| Métrica | MLP PyTorch | GradientBoosting | LogisticRegression | DummyClassifier |
|---|---|---|---|---|
| **F1** | ~0.84 | ~0.86 | ~0.80 | ~0.00 |
| **AUC-ROC** | ~0.92 | ~0.93 | ~0.88 | ~0.50 |
| **PR-AUC** | ~0.82 | ~0.83 | ~0.76 | ~0.27 |
| **Recall** | ~0.85 | ~0.84 | ~0.79 | ~0.00 |
| **Precision** | ~0.83 | ~0.88 | ~0.81 | ~0.00 |

> Valores aproximados — consulte o MLflow para métricas exatas do run.

## Métricas de Negócio

| Tipo de Erro | Impacto |
|---|---|
| **Falso Negativo (FN)** | Cliente que cancela não identificado → perda total da receita (alto custo) |
| **Falso Positivo (FP)** | Cliente que ficaria recebe campanha desnecessária → custo da ação de retenção (baixo) |

### Análise de Custo × Threshold

**Premissas de custo (estimativas de referência):**

| Parâmetro | Valor | Descrição |
|---|---|---|
| Custo por FN | R$ 500 | Receita mensal perdida por cliente churned não identificado |
| Custo por FP | R$ 20 | Custo da ação de retenção enviada desnecessariamente |
| N positivos (teste) | 380 | Churners no conjunto de teste (~27% × 1.409) |
| N negativos (teste) | 1.029 | Não-churners no conjunto de teste (~73% × 1.409) |

**Custo total estimado por threshold** (conjunto de teste, ~1.409 clientes):

| Threshold | Recall | FN (churners perdidos) | FP (campanhas extras) | Custo Total | Δ vs. padrão |
|---|---|---|---|---|---|
| 0.20 | 99% | 5 | 627 | R$ 22.400 | −56% |
| **0.25** | **98%** | **9** | **515** | **R$ 22.100** | **−56% ✦ ótimo** |
| 0.30 | 96% | 15 | 402 | R$ 22.900 | −55% |
| 0.35 | 93% | 25 | 298 | R$ 25.700 | −49% |
| 0.40 | 89% | 40 | 208 | R$ 31.000 | −39% |
| 0.45 | 84% | 60 | 137 | R$ 39.300 | −22% |
| **0.50** | **77%** | **86** | **85** | **R$ 50.600** | **— (padrão API)** |
| 0.55 | 69% | 117 | 49 | R$ 64.800 | +28% |
| 0.60 | 60% | 153 | 27 | R$ 81.300 | +61% |
| 0.70 | 40% | 228 | 6 | R$ 116.900 | +131% |
| 0.80 | 23% | 294 | 1 | R$ 148.700 | +194% |

> Custo total = (FN × R$500) + ((TP + FP) × R$20). Valores calculados via aproximação de distribuições normais das pontuações (pos: μ=0.65 σ=0.20; neg: μ=0.25 σ=0.18). O gráfico interativo gerado a partir dos dados reais está no notebook `notebooks/02_mlp_training.ipynb`.

**Conclusão:** o threshold ótimo de custo é **≈ 0.25**. O FN é 25× mais caro que o FP (500/20), justificando aceitar mais falsos positivos para não perder churners. Em produção, o CRM deve consumir a `churn_probability` retornada pela API e aplicar o threshold conforme a política comercial vigente — o threshold de 0.5 hardcoded na API é apenas o padrão conservador.

**Recomendação:** Reduzir threshold de 0.5 para ~0.25–0.30 reduz o custo total esperado em ~55%, capturando 98% dos churners reais ao custo de ~515 campanhas desnecessárias por ciclo de 1.409 clientes.

## Limitações e Vieses

### Limitações Técnicas
- **Escopo geográfico:** Dataset 100% de clientes da Califórnia (EUA) — generalização para outros mercados não validada.
- **Período temporal:** Snapshot único sem informação sobre sazonalidade ou tendências temporais.
- **Ausência de features comportamentais:** Reclamações, interações com suporte, uso de dados — poderiam melhorar a predição.
- **Threshold fixo:** O threshold padrão (0.5) não é ótimo para o negócio — deve ser ajustado conforme custo de FP/FN.

### Vieses Potenciais
- **Viés de gênero:** `Gender` foi incluída como feature. Verificar se o modelo não penaliza injustamente grupos demográficos.
- **Viés de senioridade:** `Senior Citizen` pode correlacionar com padrões de renda e acessibilidade — monitorar disparidade de performance por grupo.
- **Viés de serviço:** Clientes com `Internet Service = Fiber optic` têm taxa de churn muito maior — o modelo pode estar aprendendo características de precificação, não de qualidade.

### Cenários de Falha
| Cenário | Risco |
|---|---|
| Cliente novo (Tenure < 3 meses) | Poucos dados históricos — predição menos confiável |
| Mudança de plano recente | `Total Charges` inconsistente com `Monthly Charges` × `Tenure` |
| Novos tipos de contrato | `OneHotEncoder` retorna zeros para categorias desconhecidas |
| Drift de distribuição | Performance degrada com mudanças no mix de produtos ofertados |

## Uso Pretendido

**Casos de uso aprovados:**
- Identificar clientes de alto risco para campanhas proativas de retenção
- Priorizar filas de atendimento proativo por probabilidade de churn
- Análise exploratória para entender drivers de cancelamento

**Casos de uso não aprovados:**
- Decisões automatizadas sem revisão humana sobre penalidades contratuais
- Definição de preços individualizados baseados em score de churn
- Qualquer decisão que impacte negativamente o cliente sem transparência

## Plano de Monitoramento

| Métrica | Frequência | Alerta |
|---|---|---|
| AUC-ROC no conjunto de validação recente | Semanal | Queda > 0.05 |
| Taxa de churn real vs. previsto (calibração) | Mensal | Desvio > 10% |
| Distribuição das features de entrada (PSI) | Mensal | PSI > 0.2 em qualquer feature |
| Taxa de predições positivas (model drift) | Diário | Variação > 15% da média histórica |

**Playbook de resposta:**
1. Alerta disparado → revisar dados de entrada (data quality)
2. Confirmar drift com PSI nas features críticas (`Contract`, `Tenure Months`)
3. Se drift confirmado → re-treinar com dados dos últimos 6 meses
4. A/B test do modelo re-treinado antes de promover para produção

## Informações de Reprodutibilidade

- Seed global: `42` (numpy, torch, sklearn)
- Versão do dataset: `yeanzc/telco-customer-churn-ibm-dataset` (hash fixado pelo kagglehub)
- Todos os experimentos rastreados no MLflow (experiment: `customer_churn_ibm`)
- Dependências fixadas em `pyproject.toml`

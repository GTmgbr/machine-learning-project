# Predição de Chuva no Dia Seguinte com Machine Learning

**Disciplina:** PCO213 — Aprendizado de Máquina e Mineração de Dados  
**Instituição:** UNIFEI — Universidade Federal de Itajubá  
**Dataset:** Rain in Australia (weatherAUS) — Kaggle  

---

## 1. Visão geral do projeto

Este projeto desenvolve um pipeline completo e reproduzível de ciência de dados para **classificação supervisionada binária**: prever se choverá no dia seguinte (`RainTomorrow = Yes` ou `No`) a partir de variáveis meteorológicas observadas no dia atual.

O pipeline cobre todas as etapas canônicas: carregamento e inspeção do dataset, análise exploratória de dados (EDA), limpeza estrutural, divisão treino/teste, pré-processamento dentro de `Pipeline`/`ColumnTransformer` (sem data leakage), validação cruzada estratificada, treinamento e tuning de múltiplos modelos, avaliação final com métricas adequadas ao desbalanceamento de classes, e interpretação das variáveis mais relevantes.

Todos os artefatos gerados (12 figuras, 5 tabelas, melhor modelo exportado) estão em `outputs/` e `models/`. O código é organizado em módulos em `src/` e orquestrado por `main.py`.

---

## 2. Problema tratado

### 2.1 Definição formal

A previsão de chuva é formulada como um problema de **classificação supervisionada binária**:

- **Entrada:** vetor de atributos meteorológicos observados no dia atual (temperatura, umidade, pressão, vento, etc.).
- **Saída:** variável binária `RainTomorrow` — `Yes` (1) se haverá precipitação no dia seguinte, `No` (0) caso contrário.
- **Objetivo:** aprender uma função `f: X → {0, 1}` que generalize para dados não vistos.

### 2.2 Relevância prática

A antecipação de eventos chuvosos tem impacto direto em múltiplos domínios:

- **Agricultura:** planejamento de colheitas, irrigação e aplicação de defensivos.
- **Defesa civil:** emissão de alertas para prevenção de enchentes e deslizamentos.
- **Transportes e logística:** otimização de rotas, prevenção de acidentes.
- **Turismo e eventos:** orientação de decisões operacionais ao ar livre.
- **Planejamento urbano:** gestão de recursos hídricos e infraestrutura de drenagem.

O **falso negativo** — prever "não choverá" quando de fato choverá — tem custo prático superior ao falso positivo na maioria desses contextos, pois impede a tomada de medidas preventivas. Por essa razão, métricas como *recall* e F1-score foram priorizadas sobre a acurácia simples.

---

## 3. Dataset utilizado

### 3.1 Fonte e descrição

| Atributo | Valor |
|---|---|
| Nome | Rain in Australia / weatherAUS |
| Fonte | Kaggle — `jsphyg/weather-dataset-rattle-package` |
| Arquivo | `weatherAUS.csv` |
| Registros originais | 145.460 observações × 23 colunas |
| Período aproximado | 2007 a 2017 |
| Estações | 49 estações meteorológicas australianas |
| Variável-alvo | `RainTomorrow` — binário: `Yes` (chove) / `No` (não chove) |

Após remoção de linhas com `RainTomorrow` ausente (3.267 registros sem rótulo válido), o dataset de trabalho passou a contar com **142.193 registros**. Após limpeza estrutural (remoção de colunas com alto percentual de ausentes e engenharia de atributos de data), o dataset processado tem **142.193 linhas × 21 colunas**.

### 3.2 Distribuição da variável-alvo

| Classe | Registros | Proporção |
|---|---|---|
| `No` (não choverá) | 110.316 | 77,6% |
| `Yes` (choverá) | 31.877 | 22,4% |

O dataset apresenta **desbalanceamento de classes** significativo: a classe positiva (`Yes`) representa apenas 22,4% dos dados. Isso torna a acurácia uma métrica enganosa — um classificador que sempre prevê `No` obteria 77,6% de acurácia sem nenhum poder preditivo real. Por essa razão, F1-score e ROC-AUC foram usados como critérios primários de avaliação.

---

## 4. Pipeline desenvolvido

O fluxo completo do pipeline é:

```mermaid
flowchart LR
  A[Dataset bruto\nweatherAUS.csv] --> B[EDA\nFig 1-7]
  B --> C[Limpeza estrutural\ndata/processed/]
  C --> D[Split treino/teste\n80% / 20% estratificado]
  D --> E[Pipeline sklearn\nColumnTransformer]
  E --> F[Validacao cruzada\nStratifiedKFold k=5]
  F --> G[Tuning\nGrid/RandomizedSearch]
  G --> H[Treinamento\n5 modelos]
  H --> I[Avaliacao no teste\nFig 8-12]
  I --> J[Selecao do\nmelhor modelo]
  J --> K[Relatorio final]
```

Todas as etapas são implementadas em módulos Python em `src/` e orquestradas por `main.py`. O pipeline é integralmente reproduzível: basta executar `python main.py` com o arquivo `weatherAUS.csv` em `data/raw/`.

---

## 5. Análise exploratória dos dados (EDA)

A EDA foi conduzida em sete etapas, cada uma gerando uma figura salva em `outputs/figures/`:

| Figura | Arquivo | Conteúdo |
|---|---|---|
| Fig. 1 | `01_target_distribution.png` | Distribuição de `RainTomorrow`: 77,6% `No` vs 22,4% `Yes` |
| Fig. 2 | `02_missing_values.png` | Percentual de ausentes por coluna |
| Fig. 3 | `03_numeric_histograms.png` | Histogramas das variáveis numéricas |
| Fig. 4 | `04_numeric_boxplots.png` | Boxplots agrupados por `RainTomorrow` |
| Fig. 5 | `05_correlation_heatmap.png` | Heatmap de correlação de Pearson |
| Fig. 6 | `06_raintoday_vs_tomorrow.png` | Relação entre `RainToday` e `RainTomorrow` |
| Fig. 7 | `07_scatter_relationships.png` | Scatter plots de pares meteorológicos coloridos por `RainTomorrow` |

### 5.1 Valores ausentes

O percentual de ausentes variou de 0,87% (`MaxTemp`) a 48,0% (`Sunshine`). As colunas com mais de 40% de ausentes foram removidas na limpeza estrutural:

| Coluna | Ausentes |
|---|---|
| `Sunshine` | 48,0% |
| `Evaporation` | 43,2% |
| `Cloud3pm` | 40,8% |
| `Cloud9am` | 38,4% (mantida — abaixo do limiar) |

### 5.2 Outliers

A análise pelo método IQR (salva em `outputs/tables/outlier_analysis.csv`) identificou valores extremos em **12 das 16 colunas numéricas** analisadas. Optou-se por não remover outliers, pois podem representar eventos meteorológicos extremos de alta relevância preditiva (ex.: `Rainfall` com 17,99% de outliers inclui tempestades intensas).

### 5.3 Correlações relevantes

A análise de correlação de Pearson (Fig. 5) revelou:
- **`Humidity3pm`** apresenta correlação positiva com `RainTomorrow` (**r = 0,4462**): alta umidade relativa à tarde é um forte indicador de precipitação.
- **`Pressure3pm`** apresenta correlação negativa (**r = −0,2260**): queda de pressão à tarde sinaliza aproximação de frentes chuvosas.

Os scatter plots (Fig. 7) confirmam separação visual entre classes para os pares `Humidity3pm × Pressure3pm` e `Humidity3pm × Temp3pm`, evidenciando que alta umidade e baixa pressão à tarde caracterizam dias chuvosos.

### 5.4 Relação RainToday → RainTomorrow

A Fig. 6 mostra que dias com `RainToday = Yes` apresentam probabilidade consideravelmente maior de `RainTomorrow = Yes`, confirmando a relevância desta variável como preditor direto.

---

## 6. Pré-processamento

### 6.1 Limpeza estrutural (antes do split)

Operações aplicadas ao dataset bruto antes da divisão treino/teste:

1. Remoção de 3.267 linhas com `RainTomorrow` ausente.
2. Remoção de colunas com percentual de ausentes acima de 40%: `Evaporation` (43,2%), `Sunshine` (48,0%), `Cloud3pm` (40,8%).
3. Conversão de `RainToday` e `RainTomorrow`: `Yes → 1`, `No → 0`.
4. Extração de `Month` (mês 1–12) e `Season` (estação australiana) a partir da coluna `Date`; remoção da coluna `Date` original.

> **Nota sobre data leakage:** a remoção de colunas por alto percentual de ausentes é uma operação estrutural e determinística, baseada na taxa de missingness do dataset completo e na relevância meteorológica da variável — sem uso da variável-alvo nem de estatísticas do conjunto de teste. Não configura data leakage.

### 6.2 Pipeline/ColumnTransformer (ajustado apenas no treino)

Após o split, todas as transformações de imputação, codificação e normalização são realizadas dentro de `Pipeline` e `ColumnTransformer` do scikit-learn, **ajustados exclusivamente no conjunto de treino**:

| Tipo de variável | Transformações |
|---|---|
| Numéricas | `SimpleImputer(strategy='median')` → `StandardScaler()` |
| Categóricas | `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')` |

Esta arquitetura impede que qualquer informação do conjunto de teste contamine o treinamento.

### 6.3 Divisão treino/teste

```
train_test_split(test_size=0.2, stratify=y, random_state=42)
```

| Conjunto | Amostras | Proporção |
|---|---|---|
| Treino | 113.754 | 80% |
| Teste | 28.439 | 20% |

O conjunto de teste foi isolado e utilizado **apenas** na avaliação final, sem nenhum contato com o processo de treinamento ou tuning.

---

## 7. Modelos avaliados

Cinco algoritmos foram implementados e comparados:

| Modelo | Configuração principal | Observação |
|---|---|---|
| `DummyClassifier` | `strategy='most_frequent'` | Baseline majoritário — referência mínima |
| `LogisticRegression` | `class_weight='balanced'`, `max_iter=1000` | Baseline interpretável, coeficientes lineares |
| `DecisionTreeClassifier` | `class_weight='balanced'` | Tuning via `GridSearchCV` |
| `RandomForestClassifier` | `class_weight='balanced'`, `n_estimators=100` | Tuning via `RandomizedSearchCV(n_iter=20)` |
| `LinearSVC` (calibrado) | `class_weight='balanced'`, `max_iter=2000` | Encapsulado em `CalibratedClassifierCV(cv=3)` para `predict_proba`; treinado em amostra estratificada de 30.000 amostras do treino |

O `DummyClassifier` serve como referência: qualquer modelo útil deve superá-lo amplamente em F1 e ROC-AUC. O `LinearSVC` foi treinado em amostra por custo computacional — o `SVC` com kernel RBF tem complexidade O(n²), inviável para ~113.754 amostras.

### 7.1 Otimização de hiperparâmetros

| Modelo | Estratégia | Grade/Espaço |
|---|---|---|
| `DecisionTree` | `GridSearchCV` | `max_depth ∈ {3,5,8,None}`, `min_samples_split ∈ {2,10,20}`, `min_samples_leaf ∈ {1,5,10}`, `criterion ∈ {gini, entropy}` |
| `RandomForest` | `RandomizedSearchCV(n_iter=20)` | `n_estimators ∈ {50,100,200}`, `max_depth ∈ {5,10,20,None}`, `min_samples_split ∈ {2,5,10}`, `min_samples_leaf ∈ {1,3,5}` |
| `LogisticRegression` | `GridSearchCV` | `C ∈ {0.01, 0.1, 1.0, 10.0}` |

---

## 8. Validação e métricas

### 8.1 Validação cruzada

A seleção do melhor modelo utilizou `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` no conjunto de treino. A estratificação garante que a proporção de classes (~78/22%) seja preservada em cada fold, evitando estimativas de desempenho enviesadas.

### 8.2 Métricas utilizadas

Para um classificador binário, as métricas derivadas da matriz de confusão são:

| Métrica | Fórmula | Interpretação |
|---|---|---|
| Accuracy | (TP + TN) / total | Proporção de classificações corretas — enganosa com desbalanceamento |
| Precision | TP / (TP + FP) | Dos previstos como "choverá", quantos realmente choveram |
| Recall | TP / (TP + FN) | Dos dias em que choveu, quantos foram corretamente previstos |
| F1-score | 2 · Prec · Rec / (Prec + Rec) | Média harmônica de Precision e Recall |
| ROC-AUC | Área sob curva ROC | Capacidade discriminativa independente do limiar |
| Average Precision | Área sob curva PR | Sumariza curva Precision-Recall; mais informativa com desbalanceamento |

**Por que Accuracy não basta:** o `DummyClassifier` (que sempre prevê `No`) obteve Accuracy = 0,7758 — aparentemente alto — mas F1 = 0,0000 e ROC-AUC = 0,5000, confirmando que não tem poder preditivo algum. Este é o principal argumento para usar F1 e ROC-AUC como critérios de seleção.

---

## 9. Resultados obtidos

### 9.1 Métricas no conjunto de teste

**Tabela — Métricas finais no conjunto de teste (salva em `outputs/tables/metrics_comparison.csv`):**

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg Precision |
|---|---|---|---|---|---|---|
| **RandomForest** | **0,8483** | **0,6643** | 0,6536 | **0,6589** | **0,8868** | **0,7324** |
| LogReg | 0,7905 | 0,5221 | **0,7724** | 0,6230 | 0,8667 | 0,6989 |
| DecTree | 0,7699 | 0,4914 | 0,7587 | 0,5965 | 0,8472 | 0,6706 |
| SVM | 0,8442 | **0,7258** | 0,4904 | 0,5853 | 0,8647 | 0,6954 |
| Baseline | 0,7758 | 0,0000 | 0,0000 | 0,0000 | 0,5000 | 0,2242 |

### 9.2 Análise por modelo

**RandomForest (melhor modelo geral)**  
Obteve F1 = 0,6589 e ROC-AUC = 0,8868 — melhor em ambas as métricas primárias. Como método *ensemble* baseado em *bagging* e *feature randomness*, o Random Forest reduz a variância em relação a uma árvore única, generalizando melhor para dados não vistos. O ganho em F1 em relação ao baseline foi de **+0,6589 (≈ +65,9 pontos percentuais)**.

**Logistic Regression (maior Recall)**  
Obteve Recall = 0,7724 — o maior entre todos os modelos. Isso significa que a Regressão Logística detecta 77,24% dos dias chuvosos reais. Se o objetivo for **minimizar falsos negativos** (por exemplo, em sistemas de alerta de defesa civil), a Regressão Logística é a alternativa preferível ao RandomForest, mesmo com F1 menor (0,6230).

**SVM/LinearSVC (maior Precision)**  
Obteve Precision = 0,7258 — quando prevê chuva, acerta em 72,58% das vezes. Entretanto, o Recall muito baixo (0,4904) indica que deixa de detectar quase metade dos dias chuvosos reais. O SVM é menos adequado quando o objetivo é cobrir o maior número possível de eventos chuvosos.

**Decision Tree**  
Obteve F1 = 0,5965 após tuning via GridSearchCV — menor que a Regressão Logística (0,6230), apesar de maior Recall (0,7587). A árvore de decisão tem maior variância que métodos *ensemble*, o que explica o desempenho inferior ao RandomForest.

**Baseline (DummyClassifier)**  
Accuracy = 0,7758 com F1 = 0,0000. Demonstra que acurácia sozinha é uma métrica enganosa em datasets desbalanceados. Todo modelo útil deve superar este baseline amplamente em F1 e ROC-AUC.

### 9.3 Figuras de avaliação

| Figura | Arquivo | Conteúdo |
|---|---|---|
| Fig. 8 | `08_confusion_matrix.png` | Matriz de confusão do RandomForest |
| Fig. 9 | `09_roc_curves.png` | Curvas ROC de todos os modelos |
| Fig. 10 | `10_pr_curve.png` | Curva Precision-Recall do melhor modelo |
| Fig. 11 | `11_metrics_comparison.png` | Comparativo visual de métricas entre modelos |
| Fig. 12 | `12_feature_importance.png` | Importância das variáveis (RF + LogReg) |

---

## 10. Importância das variáveis

As importâncias foram extraídas do RandomForest (MDI — Mean Decrease in Impurity) após o treinamento final, disponíveis em `outputs/tables/top_features.csv`.

**Top 15 variáveis mais importantes (RandomForest):**

| Ranking | Variável | Importância | Interpretação meteorológica |
|---|---|---|---|
| 1 | `Humidity3pm` | 0,1478 | Umidade relativa às 15h; forte preditor direto de chuva |
| 2 | `Pressure3pm` | 0,0628 | Pressão atm. às 15h; queda indica frentes chuvosas |
| 3 | `Humidity9am` | 0,0611 | Umidade às 9h; indica condições gerais do dia |
| 4 | `WindGustSpeed` | 0,0538 | Rajada máxima; associada a sistemas de baixa pressão |
| 5 | `Pressure9am` | 0,0537 | Pressão atm. às 9h; tendência de queda sinaliza chuva |
| 6 | `Rainfall` | 0,0528 | Precipitação do dia atual; correlacionada com chuva futura |
| 7 | `Temp3pm` | 0,0523 | Temperatura às 15h; temperaturas mais frias favorecem precipitação |
| 8 | `MinTemp` | 0,0466 | Temperatura mínima; noites frias comuns em dias chuvosos |
| 9 | `MaxTemp` | 0,0456 | Temperatura máxima; valores mais baixos associados à chuva |
| 10 | `Temp9am` | 0,0419 | Temperatura às 9h; contexto térmico matinal |
| 11 | `Cloud9am` | 0,0369 | Nebulosidade às 9h; nebulosidade alta precede chuva |
| 12 | `WindSpeed3pm` | 0,0317 | Velocidade do vento às 15h; ventos fortes associados a frentes |
| 13 | `WindSpeed9am` | 0,0290 | Velocidade do vento às 9h |
| 14 | `RainToday` | 0,0269 | Choveu hoje (Yes/No); preditor direto da variável-alvo |
| 15 | `Month` | 0,0218 | Mês do ano; sazonalidade da precipitação |

As duas variáveis mais importantes — `Humidity3pm` (importância 0,1478) e `Pressure3pm` (0,0628) — são coerentes com o conhecimento meteorológico: alta umidade relativa à tarde indica ar saturado próximo ao ponto de condensação, enquanto baixa pressão atmosférica sinaliza a aproximação de frentes frias e sistemas ciclônicos associados à precipitação. As correlações de Pearson confirmam essa relevância: r = +0,4462 para `Humidity3pm` e r = −0,2260 para `Pressure3pm` com a variável-alvo.

---

## 11. Discussão

### 11.1 Por que RandomForest foi o melhor modelo

O Random Forest superou os demais modelos por três razões principais:

1. **Redução de variância por *ensemble*:** cada árvore é treinada em uma amostra bootstrap com subconjunto aleatório de atributos; a média das previsões cancela erros individuais.
2. **Robustez ao desbalanceamento:** `class_weight='balanced'` ajusta os pesos de cada classe inversamente à sua frequência, mitigando o viés para a classe majoritária.
3. **Capacidade de capturar interações não lineares:** diferente da Regressão Logística, o Random Forest modela relações complexas entre atributos sem necessidade de engenharia de features manual.

### 11.2 Trade-off Precision × Recall

O trade-off principal observado nos resultados:

- **RandomForest (F1 = 0,6589):** equilíbrio entre Precision (0,6643) e Recall (0,6536). Melhor escolha quando se quer balancear os dois tipos de erro.
- **LogReg (Recall = 0,7724, F1 = 0,6230):** prioriza detectar dias chuvosos ao custo de mais alertas falsos (Precision = 0,5221). Preferível em contextos onde o custo do falso negativo é muito alto (ex.: evacuação preventiva).
- **SVM (Precision = 0,7258, Recall = 0,4904):** quando alerta, acerta mais — mas deixa de detectar ~50% dos dias chuvosos reais. Preferível em contextos onde recursos de resposta são limitados e falsos positivos têm custo alto.

O `class_weight='balanced'` foi essencial para todos os modelos: sem ele, os classificadores tenderiam a ignorar a classe `Yes` e obter alta acurácia trivialmente.

### 11.3 Custo dos falsos negativos

O RandomForest, com Recall = 0,6536, estima-se que produza aproximadamente **2.208 falsos negativos** no conjunto de teste (dias em que choveria, mas o modelo previu `No`). Para os domínios de aplicação citados, cada falso negativo representa: colheitas expostas à chuva sem proteção, rotas de transporte não ajustadas, eventos ao ar livre não cancelados, ou recursos de defesa civil não pré-posicionados.

### 11.4 Limitações

- **Dados exclusivamente australianos:** o modelo não generaliza diretamente para outras regiões sem re-treinamento.
- **Ausentes elevados em colunas relevantes:** `Sunshine` (48%) e `Evaporation` (43%) foram removidas; poderiam ter valor preditivo adicional se disponíveis.
- **Independência temporal assumida:** o pipeline trata cada observação como i.i.d., ignorando a dependência entre dias consecutivos. Abordagens de séries temporais poderiam explorar essa estrutura.
- **SVM em amostra reduzida:** o `LinearSVC` foi treinado em 30.000 amostras do treino (de 113.754), o que pode subestimar seu potencial real.
- **Limiar de classificação fixo em 0,5:** ajustar o limiar de decisão com base na curva PR (Fig. 10) poderia melhorar o Recall do RandomForest sem comprometer muito a Precision.

---

## 12. Conclusão

Um pipeline completo e reproduzível de ciência de dados foi desenvolvido para a tarefa de previsão de chuva no dia seguinte, utilizando o dataset real *Rain in Australia* com 142.193 observações de 49 estações meteorológicas australianas.

Foram implementados e comparados cinco algoritmos de classificação supervisionada. O **RandomForest** apresentou o melhor desempenho geral:

- **F1-score = 0,6589** (ganho de +65,9 pp sobre o baseline)
- **ROC-AUC = 0,8868**
- **Recall = 0,6536**, **Precision = 0,6643**

A análise exploratória revelou que `Humidity3pm` e `Pressure3pm` são os preditores mais relevantes, alinhando-se ao conhecimento meteorológico. O uso de `class_weight='balanced'`, validação cruzada estratificada e métricas adequadas ao desbalanceamento foi essencial para uma avaliação honesta do desempenho dos modelos.

### Próximos passos

- Testar modelos de *gradient boosting* (XGBoost, LightGBM, CatBoost).
- Explorar abordagens de séries temporais (LSTM, Temporal Fusion Transformer) para capturar dependência temporal.
- Avaliar modelos por localidade geográfica.
- Ajustar o limiar de classificação com base na curva Precision-Recall.
- Calibrar probabilidades (Platt Scaling, Regressão Isotônica).
- Desenvolver API REST ou dashboard interativo (Streamlit/Gradio) para uso prático.
- Incorporar dados mais recentes via API do Bureau of Meteorology da Austrália.

---

## 13. Como executar o projeto

### 13.1 Pré-requisitos

- Python 3.9 ou superior
- Arquivo `weatherAUS.csv` em `data/raw/` (download via Kaggle ou manual — ver README.md)

### 13.2 Instalação

```bash
# Criar e ativar ambiente virtual
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 13.3 Execução

**Windows (PowerShell) — recomendado:**
```powershell
$env:PYTHONUTF8="1"; python main.py
```

**Windows (CMD):**
```cmd
set PYTHONUTF8=1 && python main.py
```

**Linux / macOS:**
```bash
python main.py
```

> **Nota sobre `PYTHONUTF8=1`:** necessário em consoles Windows com codificação cp1252 (padrão em muitas instalações Windows em português). Sem essa variável, os caracteres de box-drawing usados nos logs (`─`, `═`, `●`) causam `UnicodeEncodeError`. A variável força o Python a usar UTF-8 para saída no terminal sem alterar nenhum arquivo do projeto.

### 13.4 O que o pipeline gera

Após execução completa (~10 minutos, incluindo GridSearch e RandomizedSearch):

| Artefato | Localização | Descrição |
|---|---|---|
| 12 figuras | `outputs/figures/` | EDA (Fig. 1–7) e avaliação (Fig. 8–12) |
| `data_dictionary.csv` | `outputs/tables/` | Dicionário de dados com decisões de tratamento |
| `descriptive_stats.csv` | `outputs/tables/` | Estatísticas descritivas das variáveis numéricas |
| `outlier_analysis.csv` | `outputs/tables/` | Análise de outliers por IQR |
| `metrics_comparison.csv` | `outputs/tables/` | Métricas de todos os modelos no conjunto de teste |
| `top_features.csv` | `outputs/tables/` | Top 15 variáveis mais importantes do RandomForest |
| `best_model.joblib` | `models/` | Pipeline do melhor modelo (RandomForest) exportado |
| `weatherAUS_processed.csv` | `data/processed/` | Dataset após limpeza estrutural |

---

## 14. O que entregar ao professor

### 14.1 Incluir no ZIP

```
machine-learning-project/
├── src/                          ← código-fonte dos módulos
├── notebooks/                    ← notebooks (se existirem)
├── docs/
│   ├── RELATORIO_FINAL.md        ← este documento
│   ├── artigo_estrutura.md       ← estrutura do artigo científico
│   └── referencias.md
├── outputs/
│   ├── figures/                  ← 12 figuras geradas (versionadas)
│   └── tables/                   ← 5 tabelas CSV (versionadas)
├── requirements.txt
├── README.md
├── main.py
└── .gitignore
```

### 14.2 Não incluir no ZIP

| Item | Motivo |
|---|---|
| `.venv/` ou `venv/` | Ambiente virtual — regenerável com `pip install -r requirements.txt` |
| `__pycache__/` | Cache de bytecode Python — gerado automaticamente |
| `.ipynb_checkpoints/` | Checkpoints internos do Jupyter |
| `.git/` | Histórico de versão — irrelevante para entrega |

### 14.3 Envio opcional (se o professor pedir)

| Item | Observação |
|---|---|
| `models/best_model.joblib` | ~550 MB — grande demais para ZIP padrão; regenerável com `python main.py` |
| `data/processed/weatherAUS_processed.csv` | ~14 MB — regenerável pelo pipeline |
| `data/raw/weatherAUS.csv` | Dataset original — disponível no Kaggle |

Se necessário executar o projeto sem re-treinar, inclua `best_model.joblib` e `weatherAUS_processed.csv`.

---

*Documento gerado com base nos artefatos reais produzidos pela execução do pipeline em 24/06/2026.*

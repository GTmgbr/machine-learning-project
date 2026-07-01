# Predição de Chuva no Dia Seguinte — Rain in Australia

## Objetivo

Desenvolver um pipeline completo e reproduzível de ciência de dados para **classificação
supervisionada binária**: prever se choverá no dia seguinte (RainTomorrow = Yes/No)
a partir de variáveis meteorológicas observadas no dia atual.

## Dataset

| Atributo       | Descrição |
|----------------|-----------|
| **Nome**       | Rain in Australia / weatherAUS |
| **Fonte**      | [Kaggle — jsphyg/weather-dataset-rattle-package](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) |
| **Arquivo**    | weatherAUS.csv |
| **Registros**  | ~142 000 observações diárias de estações meteorológicas australianas |
| **Alvo**       | RainTomorrow — binário: Yes (chove) / No (não chove) |

## Estrutura do projeto

```
machine-learning-project/
├── data/
│   ├── raw/              # weatherAUS.csv (não versionado — ver abaixo)
│   └── processed/        # CSV com limpeza estrutural (não versionado — regenerável)
├── notebooks/            # 01_eda, 02_modeling, 03_results_interpretation
├── src/                  # Módulos Python do pipeline
│   ├── config.py         # Configurações globais e paths
│   ├── data_loader.py    # Carregamento, inspeção e dicionário de dados
│   ├── eda.py            # Análise exploratória e geração de figuras
│   ├── preprocessing.py  # Limpeza estrutural e ColumnTransformer
│   ├── modeling.py       # Definição e treinamento dos modelos
│   ├── evaluation.py     # Métricas, gráficos de avaliação e comparação
│   └── visualization.py  # Helpers de plotagem e salvamento de figuras
├── outputs/
│   ├── figures/          # Figuras geradas (versionadas)
│   ├── tables/           # Tabelas CSV de resultados (versionadas)
│   └── reports/          # Relatórios intermediários
├── models/               # best_model.joblib (não versionado — regenerável)
├── docs/                 # Relatório final, estrutura do artigo, referências
├── requirements.txt
├── README.md
├── .gitignore
└── main.py               # Orquestra o pipeline ponta a ponta
```

## Modelos implementados

| Modelo | Observação |
|---|---|
| DummyClassifier | Baseline majoritário |
| LogisticRegression | Baseline interpretável |
| DecisionTreeClassifier | Explicável, regras, importância |
| RandomForestClassifier | Ensemble robusto, importância |
| LinearSVC | SVM linear (amostra estratificada do treino por custo computacional) |

## Notas sobre reprodutibilidade

- random_state=42 em todos os pontos aplicáveis.
- Split treino/teste antes de qualquer ajuste de transformador.
- Imputação, encoding e scaling somente dentro de Pipeline/ColumnTransformer.
- O conjunto de teste é usado **apenas** na avaliação final.

## Resultados obtidos

Execução real com o dataset completo (142.193 registros, 5 modelos):

| Modelo | F1 | ROC-AUC | Recall | Precision | Accuracy |
|---|---|---|---|---|---|
| **RandomForest** | **0,6589** | **0,8868** | 0,6536 | 0,6643 | 0,8483 |
| LogReg | 0,6230 | 0,8667 | **0,7724** | 0,5221 | 0,7905 |
| DecTree | 0,5965 | 0,8472 | 0,7587 | 0,4914 | 0,7699 |
| SVM | 0,5853 | 0,8647 | 0,4904 | **0,7258** | 0,8442 |
| Baseline | 0,0000 | 0,5000 | 0,0000 | 0,0000 | 0,7758 |

**Melhor modelo: RandomForest** — F1 = 0,6589, ROC-AUC = 0,8868.  
Variáveis mais importantes: Humidity3pm (0,1478), Pressure3pm (0,0628).

Cinco algoritmos de classificação supervisionada foram implementados, otimizados e comparados com o uso de métricas de avaliação adequadas para conjuntos de dados desbalanceados, incluindo Precisão, *Recall*, pontuação F1, ROC-AUC e Precisão Média.

Entre os modelos avaliados, o Random Forest apresentou o melhor desempenho geral, alcançando um F1-score de 0,6589 e uma ROC-AUC de 0,8868. A análise de importância das variáveis ​​identificou *Humidity3pm* e *Pressure3pm* como os preditores mais relevantes, o que é consistente com o conhecimento meteorológico estabelecido sobre as variações de umidade atmosférica e pressão que antecedem eventos de precipitação.

<br>

![Image](https://github.com/user-attachments/assets/4663a5e1-daa1-4ee3-8356-600d1aa5e636)

<br>

![Image](https://github.com/user-attachments/assets/245d1101-3a62-41a6-a58f-4fb597c98015)

<br>

![Image](https://github.com/user-attachments/assets/cda2ff95-7096-42fb-b33f-cc18d5fa3187)

<br>

![Image](https://github.com/user-attachments/assets/b0696352-472c-432c-b8e1-5f379a564d5e)

<br>

![Image](https://github.com/user-attachments/assets/3e72eabf-40ad-4976-af50-57eeea70ba47)

<br>

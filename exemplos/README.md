# Exemplos

Quatro notebooks que mostram as bibliotecas em uso, executados em 2021 no Windows (Python 3.7, Anaconda). As saídas são as originais.

Em 2026, os quatro rodaram de novo de ponta a ponta em Python 3.11, com as versões de [`requirements.txt`](../requirements.txt).
- Os resultados que não dependem de sorteio bateram com os de 2021.
- No exemplo de churn, os do CatBoost mudaram na quarta casa decimal, porque a versão do CatBoost é outra.

| Notebook | Nome original | O que mostra |
|---|---|---|
| [`01_titanic_visao_geral`](01_titanic_visao_geral.ipynb) | `Exemplos_Bibliotecas` | Visão geral de todos os módulos na base do Titanic: análise exploratória (`DistribuicoesDataset`), tratamento (`TrataDataset`), o classificador e o regressor próprios (`SeriesMultiGradient*`), avaliação completa em treino e teste, inclusive por faixa de sexo, idade e tarifa, importância das variáveis e mapas de calor por passageiro |
| [`02_discretizacao_ilustrada`](02_discretizacao_ilustrada.ipynb) | `Ilustracao_Algoritmo_CortaIntevalos` | Os três algoritmos de discretização passo a passo, com as animações de [`imagens/`](imagens/): faixas de quantidade parecida, faixas que maximizam o ganho de informação e faixas que maximizam o R². Também compara o ganho de informação e o R² no treino e no teste conforme o número de faixas |
| [`03_house_prices_distribuicao`](03_house_prices_distribuicao.ipynb) | `Teste_Estimar_Distribuicao` | Previsão do preço de casas de duas formas: um regressor e a estimativa da distribuição do preço, com um classificador por faixa (`DiscretizaAlvoRegressivo` e `AvaliaDatasetsDistribuicaoProbabilidade`) |
| [`04_churn_bancario`](04_churn_bancario.ipynb) | `Modelo_Churn` | Churn de clientes de um banco: análise exploratória, o modelo próprio e o CatBoost avaliados com as bibliotecas e mapas de calor. A divisão "ingênua" entre treino e validação é comparada com uma divisão por sobrenome, que mantém clientes de mesmo sobrenome do mesmo lado |

## Como rodar

1. Instale o ambiente como no [README principal](../README.md#instalação). Os notebooks precisam também do Jupyter, do `xlrd` e do CatBoost, que já estão em `requirements.txt`.
2. Baixe os dados para `exemplos/datasets/`, com os nomes abaixo. A pasta está no `.gitignore`.

   | Arquivo | Fonte |
   |---|---|
   | `train_titanic.csv` | `train.csv` da competição [Titanic](https://www.kaggle.com/c/titanic/data) do Kaggle (`kaggle competitions download -c titanic`) |
   | `titanic_full.xls` | `titanic3.xls`, a base completa de 1.309 passageiros do [Vanderbilt Biostatistics](https://hbiostat.org/data/) |
   | `train_house.csv` | `train.csv` da competição [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) do Kaggle |
   | `Churn_Modelling.csv` | Dataset [Churn Modelling](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) do Kaggle |

3. Abra o Jupyter a partir desta pasta (`cd exemplos && jupyter notebook`), para que os caminhos `datasets/...` e `imagens/...` funcionem.

A primeira célula de cada notebook recarrega os módulos com `importlib.reload`, como era feito durante o desenvolvimento. O notebook 02 regrava as animações em `imagens/`.

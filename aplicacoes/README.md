# Aplicações

Notebooks em que usei as bibliotecas em competições do Kaggle e em estudos de redes neurais. Todos rodaram no Google Colab entre 2021 e 2022, com as bibliotecas e a chave da API do Kaggle numa pasta do Google Drive. As saídas são as originais.

Eles não foram executados de novo em 2026:
- as versões de pacotes eram as do Colab da época (Python 3.7);
- os notebooks de 2021 usam a versão das bibliotecas daquele momento, e alguns parâmetros mudaram depois. O `frac_cat` de `AvaliaDatasetsClassificacao`, por exemplo, não existe na versão final.

## Porto Seguro Data Challenge (2021)

Competição da Porto Seguro no Kaggle: [porto-seguro-data-challenge](https://www.kaggle.com/c/porto-seguro-data-challenge). Os dados exigem aceitar as regras da competição.

| Notebook | Nome original | O que faz |
|---|---|---|
| [`01_baseline`](kaggle_porto_seguro_2021/01_baseline.ipynb) | `PortoSeguro_Baseline_Ale` | Baseline: análise exploratória (`DistribuicoesDataset`), o classificador próprio (`SeriesMultiGradientClassifier`), avaliação em treino e teste (`AvaliaDatasetsClassificacao`) e submissão |
| [`02_selecao_de_features_e_aumentacao`](kaggle_porto_seguro_2021/02_selecao_de_features_e_aumentacao.ipynb) | `PortoSeguro_FeatureSelection_CrossVal_Augm_Ale3` | Versão final, com CatBoost: tratamento dos nulos, um teste separado desde o início, filtro de variáveis instáveis entre os conjuntos, validação cruzada para escolher o número de iterações, nova seleção de variáveis, aumentação de dados, avaliação e importância das variáveis com as bibliotecas, e submissão |

## Ubiquant Market Prediction (2022)

Competição de previsão de retornos no Kaggle: [ubiquant-market-prediction](https://www.kaggle.com/c/ubiquant-market-prediction). Os dois notebooks importam as bibliotecas, mas o modelo e a avaliação usam o CatBoost, o Keras e funções escritas no próprio notebook.

| Notebook | Nome original | O que faz |
|---|---|---|
| [`01_baseline`](kaggle_ubiquant_2022/01_baseline.ipynb) | `baseline` | Análise exploratória com dependência temporal e um teste separado no tempo e por ativo (`investment_id`). Validação cruzada em janelas de tempo, filtro de variáveis com potencial de overfitting e um `CatBoostRegressor` com o número de iterações escolhido na validação. A avaliação usa a correlação média por instante, a métrica da competição |
| [`02_rede_neural`](kaggle_ubiquant_2022/02_rede_neural.ipynb) | `rede_neural` | O mesmo desenho de validação, com redes neurais densas (32-16-8-4-2 neurônios): uma por divisão da validação cruzada e uma final |

## Redes neurais (2021)

| Notebook | Nome original | O que faz |
|---|---|---|
| [`autoencoder_fraude_cartao`](redes_neurais_2021/autoencoder_fraude_cartao.ipynb) | `Arquitetura_RedeAutoencoder` | Autoencoder na base pública de fraudes em cartão de crédito da ULB, hoje no Kaggle como [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). O notebook baixava uma cópia do datahub.io que saiu do ar. Etapas: espaço latente, agrupamento e detecção de anomalias, avaliados contra o rótulo de fraude, que fica oculto no treino |
| [`rede_recorrente_atividades`](redes_neurais_2021/rede_recorrente_atividades.ipynb) | `Arquitetura_RedeRecorrente` | Rede recorrente (`SimpleRNN`) para reconhecer atividades físicas pelos sinais do acelerômetro do celular, na base [WISDM](https://www.cis.fordham.edu/wisdm/dataset.php) |

## Para rodar hoje

- **Bibliotecas.**
  - No Colab, ponha os arquivos de [`src/`](../src/) numa pasta e aponte `path_notebook` para ela, ou clone este repositório e use `src/`.
  - Os dois notebooks de redes neurais clonavam o repositório antigo das bibliotecas, `alexandrebcruz/MinhasLibs`, hoje privado. Troque essa linha pelo clone deste repositório.
- **Chave do Kaggle.** Use a sua, em `~/.kaggle/kaggle.json`. Nunca a deixe na pasta do projeto nem a versione: o `.gitignore` já bloqueia o arquivo.
- **Versões.** O código das bibliotecas pede as versões de [`requirements.txt`](../requirements.txt). O Colab de hoje usa versões mais novas, então o mais simples é rodar num ambiente local com essas versões.

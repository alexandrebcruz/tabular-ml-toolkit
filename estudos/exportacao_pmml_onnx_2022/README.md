# Exportação de modelos para PMML e ONNX (2022)

Quatro notebooks que levam um modelo treinado em Python para formatos usados em produção fora do Python:
- **PMML**, um XML que motores de decisão e plataformas de scoring executam;
- **ONNX**, um formato aberto com runtime em várias linguagens.

A base é o Titanic, baixado de `storage.googleapis.com/tf-datasets/titanic/train.csv`. As saídas são as originais.

| Notebook | Modelo | Formato |
|---|---|---|
| [`SklearnTree_para_PMML`](SklearnTree_para_PMML.ipynb) | Uma árvore de decisão do scikit-learn. O PMML gerado é ajustado à mão (`selectFirst`) para conter só ela | PMML (`sklearn2pmml`, executado com `pypmml`) |
| [`EnsembleSklearnTree_para_PMML`](EnsembleSklearnTree_para_PMML.ipynb) | Ensemble de 10 árvores, uma por divisão de treino e validação | PMML |
| [`EnsembleXGBoost_para_PMML`](EnsembleXGBoost_para_PMML.ipynb) | Ensemble de 10 XGBoosts | PMML |
| [`EnsembleXGBoost_para_ONNX`](EnsembleXGBoost_para_ONNX.ipynb) | Ensemble de 10 XGBoosts | ONNX (`skl2onnx` com o conversor do `onnxmltools`, executado com `onnxruntime`) |

**Roteiro de cada notebook:**
- separa 10% para teste e sorteia 10 divisões de treino e validação (75% e 25%);
- em cada divisão, treina um modelo e escolhe a profundidade da árvore, ou o número de árvores do XGBoost, pela curva de viés e variância (LogLoss no treino e na validação);
- nos três ensembles, junta os modelos já treinados num `VotingClassifier` de voto suave, sem treinar de novo;
- exporta o modelo, executa o arquivo exportado e confere que as probabilidades são as mesmas do Python;
- avalia o KS no teste.

O truque para juntar modelos já treinados é ajustar o `VotingClassifier` numa amostra pequena e depois trocar os estimadores pelos treinados. Isso vale para os dois formatos.

**Para rodar:** `sklearn2pmml` precisa de Java instalado. Os pacotes são `scikit-learn`, `xgboost`, `sklearn2pmml`, `pypmml`, `skl2onnx`, `onnxmltools` e `onnxruntime`. Os notebooks rodaram em 2022 com as versões daquela época.

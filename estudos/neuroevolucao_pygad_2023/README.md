# Neuroevolução com PyGAD (2023)

Um estudo curto, de novembro de 2023: treinar uma rede neural pequena por algoritmo genético, sem gradiente. Usa a biblioteca [PyGAD](https://pygad.readthedocs.io/) (`pygad.kerasga`) com o TensorFlow. O notebook rodou no Google Colab com GPU, e as saídas são as originais.

## Dados

São as faixas do Spotify do [TidyTuesday de 21/01/2020](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-01-21).
- **Alvo:** a popularidade da faixa (`track_popularity`), dividida por 100.
- **Entradas:** 12 atributos de áudio padronizados, como dançabilidade, energia, volume, valência e andamento.

## O que o notebook faz

1. Reproduz o exemplo de início rápido da documentação do PyGAD: achar os pesos de uma combinação linear que dê 44.
2. Monta uma rede densa 12-8-4-2-1, com ReLU e saída sigmoide, num total de 153 pesos.
3. Evolui esses pesos com o algoritmo genético:
   - população de 20 redes;
   - 10 pais por geração;
   - 50 gerações.
4. Mede o *fitness* como 1 − EQM/variância, isto é, o R² na própria base.

**Avaliação da população na GPU.** O PyGAD, por padrão, avalia uma rede por vez. A subclasse `GPU_GA` troca `cal_pop_fitness`: ela monta um único modelo Keras com as 20 redes da população lado a lado, sobre uma entrada compartilhada. Assim, cada geração é avaliada numa só chamada de `predict`.

## Resultado

O melhor *fitness* subiu de −0,17 para 0,00001 em 50 gerações. Ou seja, a rede só alcançou o modelo constante, que prevê a popularidade média. Com esse orçamento de busca, ela não aprendeu nada além da média. Também não está claro que os atributos de áudio sozinhos expliquem a popularidade. O notebook não separa treino e teste, porque o objetivo era testar o método, e não o modelo.

## Para rodar

- Baixe o `spotify_songs.csv` do link acima para uma pasta `dados/`.
- Troque `<PASTA_DO_PROJETO>` pela pasta que contém `dados/`. No original, era uma pasta do Google Drive montada no Colab.
- Pacotes: `pygad` (o estudo usou a 3.2.0), `tensorflow`, `pandas` e `matplotlib`.

| Notebook | Nome original |
|---|---|
| [`rede_neural_por_algoritmo_genetico`](rede_neural_por_algoritmo_genetico.ipynb) | `Algoritmo_Genetico` |

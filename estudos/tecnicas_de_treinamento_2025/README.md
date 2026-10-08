# Técnicas de treinamento (2025)

Este estudo compara formas de escolher e treinar um modelo de classificação, com um protocolo de validação que inclui uma base que nenhuma etapa do desenvolvimento enxerga. As perguntas são três:
1. **Busca de hiperparâmetros.** Grid search ou um ensemble guiado pela curva de viés e variância?
2. **Segmentação.** Um modelo único ou um modelo por segmento?
3. **Tamanho da amostra.** Quanto dado de treino é preciso?

A base é a pública [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), com 7.043 clientes e uma taxa de churn de 26,5%. Os notebooks rodaram no Google Colab em setembro de 2025, e as saídas são as originais.

## Desenho

- **`T0_Cria_Dataset_Oculto`.**
  - Separa metade da base, sorteada, como **base oculta**, guardada até o fim. A outra metade é a base de desenvolvimento.
  - O notebook também testa, em código desativado, uma base oculta formada por um agrupamento (k-means) e por anomalias (Isolation Forest), para simular uma população diferente.
- **Divisão em cada notebook.** A base de desenvolvimento é dividida ao meio, em treino e teste, e cada modelo é avaliado em AUC e KS:
  - no treino, no teste e na base oculta;
  - em dois segmentos: clientes sem parceiro e com parceiro, chamados de "solteiros" e "casados" nos notebooks.
- **Os quatro desenhos:**

  | Notebook | Busca de hiperparâmetros | Modelagem |
  |---|---|---|
  | `T1_GridSearch_SemSplit` | Grid search de um LightGBM (taxa de aprendizado, profundidade e número de árvores), com validação cruzada estratificada de 4 dobras e AUC como critério | Modelo único |
  | `T2_GridSearch_ComSplit` | Igual ao T1 | Um modelo por segmento |
  | `T3_CurvaViesVariancia_SemSplit` | 10 CatBoosts de profundidade 1, cada um treinado numa metade sorteada do treino e parado no melhor ponto da outra metade, a validação dele. O modelo final é a média dos 10 | Modelo único |
  | `T4_CurvaViesVariancia_ComSplit` | Igual ao T3 | Um modelo por segmento |

- **Tamanho da amostra.**
  - `T5_1` e `T6` repetem o T3 e o T1 com frações de 5% a 100% do treino.
  - `T5_2` repete o T5_1 com só 6 das 30 variáveis.
- **Extras em cada notebook:**
  - calibração por faixa de probabilidade, com `CortaIntervalosQuasiUniforme`;
  - uma escala de risco com o maior número de faixas em que a taxa de churn sobe faixa a faixa, com `CortaIntervalosGanhoInformacao`.
- **Vazamento.** A opção `usar_teste_treinamento` simula vazamento, deixando o teste entrar na escolha dos hiperparâmetros. As saídas salvas são sem vazamento.

`AleFuncoes.py` traz uma versão enxuta das funções de discretização das bibliotecas, só com numpy e pandas, sem numba. Por isso roda com as versões atuais dos pacotes.

## Resultados

AUC e KS em %, com o treino completo (fração 100%):

| Notebook | AUC treino | AUC teste | AUC oculta | KS treino | KS teste | KS oculta |
|---|---|---|---|---|---|---|
| T1: grid search, modelo único | 87,65 | 83,55 | 85,00 | 59,94 | 52,52 | 55,03 |
| T2: grid search, por segmento | 87,40 | 82,01 | 84,05 | 58,20 | 49,93 | 53,14 |
| T3: curva viés-variância, modelo único | 84,72 | **84,15** | **85,66** | 53,12 | **53,58** | **56,01** |
| T4: curva viés-variância, por segmento | 85,19 | 83,62 | 84,83 | 56,76 | 53,42 | 53,88 |

1. **O grid search superestima o modelo.**
   - No T1, o AUC cai 4,1 pontos do treino para o teste.
   - O ensemble do T3 quase não perde (0,6 ponto) e tem o melhor AUC e o melhor KS no teste e na base oculta.
2. **Separar por segmento não ajudou.** Nos dois métodos, o modelo único foi melhor que os modelos por segmento, no teste e na base oculta.
3. **Com o ensemble, 40% do treino basta.**
   - No T5_1, o AUC no teste sobe de 79,0 (5% do treino) para cerca de 84 a partir de 40% e depois fica estável.
   - A diferença entre treino e teste cai de 11,8 pontos para cerca de 1 ponto ou menos.
   - Com grid search (T6), o AUC no teste foi menor em todas as frações, e a diferença entre treino e teste oscila conforme os hiperparâmetros escolhidos: de quase zero a 14 pontos.
4. **Poucas variáveis perdem pouco.** Com só 6 variáveis (T5_2), o AUC no teste com o treino completo foi de 83,7, contra 84,2 com as 30.

Os números são de uma execução, numa base pequena: as conclusões valem como indício.

## Para rodar

1. Baixe a base para esta pasta: `kaggle datasets download -d blastchar/telco-customer-churn`, que gera `telco-customer-churn.zip`.
2. Rode o `T0`, que grava `telco-customer-churn-real.zip` e `telco-customer-churn-oculto.zip`.
3. Rode os outros notebooks, que leem esses dois arquivos e importam `AleFuncoes.py` desta pasta.
4. Pacotes: pandas, scikit-learn, scipy, matplotlib, LightGBM e CatBoost, em versões atuais. Os `.zip` estão no `.gitignore`.

# Treinamento personalizado de redes neurais (2024–2025)

Um otimizador próprio para redes neurais, que leva a elas a ideia do `SeriesMultiGradient` das bibliotecas: um passo de treino só vale se melhorar ao mesmo tempo todos os subconjuntos do treino. O teste é no Titanic, do Kaggle, com um CatBoost como comparação. Os notebooks rodaram no Google Colab com GPU entre julho de 2024 e janeiro de 2025, e as saídas são as originais.

## O algoritmo

1. **Subconjuntos.** O treino é dividido em subconjuntos: 6 no notebook 01 e 2 no 02.
2. **Gradientes por subconjunto.** A cada época, calculam-se, em cada subconjunto:
   - a perda e o seu gradiente;
   - o gradiente da norma ao quadrado do gradiente, por produtos hessiana-vetor.
3. **Direção comum.** Um problema convexo, resolvido com o [cvxpy](https://www.cvxpy.org/), escolhe a direção de busca. É um vetor de norma no máximo 1 que reduz a perda e a norma do gradiente em todos os subconjuntos, com a menor variância possível entre essas reduções.
4. **Tamanho do passo.** Uma busca linear do tipo Hager-Zhang, modificada e com interpolação cúbica, escolhe o passo. A condição de Armijo é exigida em todos os subconjuntos, e não só na média.
5. **Parada.** O treino para quando a perda deixa de cair mais que um mínimo (`delta_loss_min`) durante várias épocas seguidas (`patience`).
6. **Ensemble.**
   - São treinadas 32 redes com sementes diferentes.
   - Uma camada de pesos não negativos que somam 1, o "ponderador", aprende a combinação delas.
   - As redes com peso zero ficam de fora.

A rede é pequena: camadas densas de 16, 8, 4 e 2 neurônios com tangente hiperbólica e saída softmax, com 476 parâmetros. As entradas são 17 variáveis padronizadas, como idade, tarifa, classe, sexo, título no nome e porto de embarque. A perda é a entropia cruzada, sempre dividida pela perda do modelo constante, que prevê a taxa média de sobrevivência para todos.

## Notebooks

| Notebook | Nome original | O que faz |
|---|---|---|
| [`01_ensemble_sobrevivencia_e_idade`](01_ensemble_sobrevivencia_e_idade.ipynb) | `TreinamentoPersonalizado_Titanic_Testes_v6` | 6 subconjuntos. Treina o ensemble para a sobrevivência e, com o mesmo algoritmo, uma regressão da idade |
| [`02_comparacao_com_catboost`](02_comparacao_com_catboost.ipynb) | `TreinamentoPersonalizado_Titanic_Testes_Comparativo_v1` | 2 subconjuntos. Compara com um CatBoost de profundidade 1, treinado em cada metade com a outra como validação, e gera as previsões para a submissão ao Kaggle |

As funções de treino (`fit_modelo` e `ensemble_treinamento`) estão dentro dos notebooks. Os dois também salvavam cada treino num bucket do Google Cloud Storage, para retomar depois de uma queda do Colab. Esse passo é opcional (`bucket = None`).

## Resultados

São os 891 passageiros do `train.csv`: os últimos 300 formam o teste, e os 591 restantes, o treino. A perda é a entropia cruzada dividida pela do modelo constante.

| Modelo | Perda no treino | Perda no teste | Acurácia no treino | Acurácia no teste |
|---|---|---|---|---|
| Rede, 6 subconjuntos (01): ensemble de 3 redes | 0,596 | 0,608 | 83,4% | 84,0% |
| Rede, 2 subconjuntos (02): ensemble de 2 redes | 0,573 | 0,613 | **84,3%** | **84,3%** |
| CatBoost de profundidade 1 (02) | — | — | 82,2% | 82,7% |

1. **Treino e teste ficaram próximos.**
   - A perda no teste foi só 2% a 7% maior que no treino.
   - A acurácia foi a mesma ou maior no teste.
   - É o comportamento que o critério de "melhorar todos os subconjuntos" busca: não decorar um pedaço do treino.
2. **A rede passou o CatBoost raso.** No teste, foram 84,3% de acurácia contra 82,7%. O CatBoost era só um termo de comparação e não foi ajustado.

**Ressalvas:**
- O teste foi o mesmo em todas as versões do estudo, então também influenciou as escolhas ao longo delas.
- Na regressão da idade, as idades faltantes, preenchidas com a média, continuam no alvo, e a variável `Tem_Age` está entre as entradas. Por isso, esse resultado (perda de 0,524 no treino e 0,650 no teste, com 7 redes) só mostra que o mesmo treino funciona em regressão.
- Os números são de uma execução, numa base pequena.

## Para rodar

1. Use o Google Colab com GPU, como no estudo, ou um ambiente com TensorFlow e Keras 3. Os pacotes são `tensorflow`, `cvxpy`, `catboost`, `pandas`, `matplotlib` e `seaborn`, e o `kaggle` para baixar os dados.
2. Ponha a sua chave da API do Kaggle (`kaggle.json`) na pasta. As primeiras células baixam a competição `titanic`.
3. Para não usar o Google Cloud Storage, troque a célula das credenciais por `bucket = None`. Para usar, faça o upload do `credentials.json` de uma conta de serviço e preencha `<PROJETO_GCP>` e `<BUCKET_GCS>`.
4. O treino de 32 redes leva bastante tempo, mesmo com GPU.

## Histórico

O estudo começou em julho de 2024 com um protótipo. Nele, um passo do L-BFGS era dado na direção comum. Depois vieram seis versões de testes e três comparativas. Daí saíram os dois notebooks publicados, a sexta de testes e a primeira comparativa. As outras ficaram de fora porque têm erros, resultados desatualizados ou não terminaram. A última comparativa, de fevereiro de 2025, testava uma direção no estilo do MGDA (*multiple gradient descent algorithm*): a perda do treino caiu até 0,42, mas a do teste subiu, e ela não chegou a ser concluída.

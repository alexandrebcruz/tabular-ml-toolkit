# Referência dos módulos

O código das classes não tem docstrings. Esta página descreve o que cada uma faz, os parâmetros principais e os métodos que os notebooks de [`exemplos/`](../exemplos/) usam. Os nomes estão em português, como no código.

**Convenções gerais**
- As classes recebem arrays do numpy ou DataFrames do pandas e calculam tudo no construtor.
- Os métodos `valor_*`, `info_*` e `retorna_*` devolvem números ou tabelas.
- Os métodos `curva_*` devolvem os pontos de um gráfico, e os `grafico_*` desenham com matplotlib e seaborn, na paleta `colorblind`, e chamam `plt.show()`. Foram feitos para Jupyter.
- `num_div` é o número de faixas das variáveis numéricas, e `num_cat` é o número de categorias mantidas nas categóricas. As demais categorias viram "resto".
- `unit` é a unidade das datas nos rótulos, como em `np.datetime_as_string` (por exemplo, `"D"`).

## `AleTransforms`: discretização e tratamento de variáveis

**`CortaIntervalosQuasiUniforme(vetor, num_div, eh_temporal=False, unit=None)`**
- Divide `vetor` em `num_div` faixas com quantidades parecidas, sem separar valores repetidos.
- O algoritmo parte de faixas com o mesmo número de valores distintos e desloca as fronteiras uma a uma enquanto o desvio-padrão das quantidades cai.
- Aceita datas, com `eh_temporal=True`.
- Métodos:
  - `aplica_discretizacao(vetor, usar_ponto_medio=False)` devolve o índice da faixa, ou o ponto médio dela, de cada valor. `NaN` continua `NaN`.
  - `info_discretizacao()` devolve a tabela `QTD`, `Min`, `Max`, `Meio`, `Str`.
  - `grafico_densidade()` e `grafico_distribuicao(bins)` desenham a distribuição.

**`CortaIntervalosGanhoInformacao(vetor, alvo, num_div, qtd_min=0, eh_temporal=False, unit=None, balancear=False)`**
- Discretização supervisionada para alvo binário (0/1): as fronteiras se deslocam enquanto a entropia ponderada das faixas cai, ou seja, enquanto o ganho de informação sobe.
- `qtd_min` é o mínimo de observações por faixa.
- `balancear=True` usa a entropia balanceada, em que as duas classes pesam igual.
- Métodos:
  - `info_discretizacao()` inclui `QTD_Alvos`.
  - `ganho_informacao()` e `ganho_informacao_balanceado()` devolvem o ganho na base de construção.
  - `calcula_ig(vetor, alvo)` mede o ganho das mesmas faixas em outra base.
  - `grafico_entropia()` desenha a frequência do alvo e a entropia por faixa.

**`CortaIntervalosR2(vetor, alvo, num_div, qtd_min=0, eh_temporal=False, unit=None, balancear=False)`**
- A mesma busca, para alvo contínuo: as fronteiras se deslocam enquanto o erro quadrático em torno da média de cada faixa cai.
- Métodos: `r2()`, `r2_balanceado()`, `calcula_r2(vetor, alvo)` e `grafico_mse_normalizado()`.

**`FiltraCategoriasRelevantes(vetor, num_cat)`**
- Mantém as `num_cat` categorias mais frequentes.
- `aplica_filtro_categorias(vetor, considera_resto=True, usar_str=False)` devolve códigos numéricos, ou o texto com `usar_str=True`. As demais categorias viram "resto" (código 0), ou `NaN` com `considera_resto=False`.

**`TrataDataset(df, num_div=20, num_cat=5, unit=None, features_numericas=None, features_categoricas=None, features_temporais=None, autorun=True)`**
- Prepara um DataFrame inteiro:
  - variáveis numéricas e temporais com mais de `num_div` valores distintos ganham uma `CortaIntervalosQuasiUniforme`;
  - variáveis categóricas ganham uma `FiltraCategoriasRelevantes`.
- Com `num_div=None`, as numéricas ficam como estão.
- Métodos:
  - `cria_one_hot_encoder(df)` prepara o one-hot das categóricas;
  - `aplica_transformacao_ohe(df, usar_ponto_medio=False)` devolve a base transformada, com o one-hot e, se pedido, com cada numérica trocada pelo ponto médio da sua faixa;
  - `trata_coluna(df, coluna, parametros_padrao=False, num_div=..., num_cat=...)` refaz o tratamento de uma coluna com outros parâmetros.

**`TaylorLaurentSeries(laurent=False, ordem=2, apenas_interacoes=False, features_numericas=None)`**
- `aplica_transformacao(df)` acrescenta as potências das variáveis até `ordem`, as potências negativas se `laurent=True`, e os produtos entre variáveis.
- Com `apenas_interacoes=True`, entram só os produtos.
- `variaveis_criadas()` lista as colunas novas.

## `AleDatasetAnalysis`: análise exploratória

**`DistribuicoesDataset(df, num_div=None, num_cat=None, unit=None, autorun=True)`**
- Resume cada coluna: tipo, nulos e faixas.
- Métodos:
  - `grafico_distribuicao(col_ref=[])` e `grafico_densidade(col_ref=[])` desenham uma coluna, várias ou todas;
  - `info_distribuicao(col)` devolve a tabela de faixas;
  - `retorna_qtds_na()` devolve os nulos por coluna.

**`AvaliaDatasetsDistribuicoes(dict_dfs, num_div=None, num_cat=None, unit=None, autorun=False)`**
- Compara as mesmas colunas em várias bases, como `{"Sobreviventes": ..., "Não sobreviventes": ...}` ou treino e teste, e desenha a distribuição de cada base no mesmo gráfico.
- Use `trata_coluna(dict_dfs, col)` antes dos gráficos.

## `AleMetrics`: métricas

**`AletricasClassificacao(y, y_prob, num_div=None, p_corte=None, p01_corte=[0, 0], p_ref=None)`**
- Recebe o alvo binário e a probabilidade prevista.
- `num_div` agrupa as probabilidades em faixas antes das contas, o que é útil em bases grandes.
- `p_corte` e `p01_corte` fixam os cortes de decisão. Sem eles, valem os cortes que maximizam o ganho de informação.
- `p_ref` é a taxa de referência das métricas `_ref`. Na comparação entre bases, é a taxa do treino.
- Métodos:
  - `valor_metricas()` devolve as métricas abaixo;
  - `calcula_matriz_confusao(p0, p1)` devolve a matriz de confusão para os cortes dados;
  - `grafico_roc()`, `grafico_ks()`, `grafico_revocacao()`, `grafico_informacao()` e `grafico_informacao_2d()` desenham as curvas.

| Métrica | Significado |
|---|---|
| `QTD`, `QTD_0`, `QTD_1`, `Frac_0`, `Frac_1` | Tamanho da base e de cada classe |
| `Soma_Prob`, `Media_Prob` | Soma e média das probabilidades: dá para comparar com `QTD_1` e `Frac_1` e ver a calibração geral |
| `LogLoss` | Entropia cruzada |
| `CoefLogLoss` | `1 − LogLoss / LogLoss` de um modelo constante igual à taxa da própria base (0 = não melhora o modelo constante, 1 = perfeito) |
| `CoefLogLoss_ref` | O mesmo, com a taxa de referência `p_ref` |
| `AUC`, `KS` | Área sob a curva ROC e distância máxima entre as distribuições acumuladas das duas classes |
| `LiftV_10`, `LiftV_20` | Fração dos positivos capturada nos 10% e 20% de maior probabilidade |
| `AlavV_10`, `AlavV_20` | Alavancagem: essa fração dividida pela fração da base selecionada (1 = sorteio) |
| `LiftF_10`, `LiftF_20`, `AlavF_10`, `AlavF_20` | O mesmo para os negativos, nos 10% e 20% de menor probabilidade |
| `IG` | Ganho de informação normalizado (fração da entropia inicial que some) do melhor corte único na probabilidade |
| `Prob_Corte` | O corte que dá esse ganho |
| `IG_2D` | Ganho de informação com dois cortes, `Prob0_Corte` < `Prob1_Corte`, que deixam no meio uma zona de incerteza |
| `Frac_Incerto_2D` | Fração da base na zona de incerteza |
| `Acurácia`, `Acurácia_Balanceada`, `TVN`, `TVP` | Medidas no corte único (TVN e TVP são as taxas de verdadeiros negativos e positivos) |
| `Acurácia_2D`, `Acurácia_Balanceada_2D`, `TVN_2D`, `TVP_2D` | As mesmas medidas fora da zona de incerteza |
| `Acurácia_Balanceada_Cond` | Média de `P(0|0)` e `P(1|1)` |
| `P(0|0)`, `P(1|1)` | Precisão de cada classe: a chance de o modelo acertar quando diz 0 ou quando diz 1 |
| `P_2D(…)`, `Acurácia_Balanceada_Cond_2D` | O mesmo com os dois cortes |

**`AletricasRegressao(y, y_pred, y_ref=None, y2_ref=None, num_kendalltau=10000)`**
- `y_ref` e `y2_ref` são a média e a média dos quadrados do alvo numa base de referência, normalmente o treino.
- `valor_metricas()` devolve:

| Métrica | Significado |
|---|---|
| `MAE`, `MSE`, `RMSE` | Erros absoluto médio, quadrático médio e a raiz dele |
| `RAE`, `RSE`, `RRSE` | `MAE` e `MSE` divididos pelos da média da própria base, e a raiz de `RSE` |
| `R1`, `R2`, `RR2` | `1 − RAE`, `1 − RSE` (o R² usual) e `1 − RRSE` |
| `…_ref` | As mesmas medidas em relação à média de referência `y_ref`, e não à da própria base |
| `KendallTau`, `KendallTau_Conc`, `KendallTau_Disc` | Concordância de ordem entre alvo e previsão, estimada com `num_kendalltau` pares sorteados |
| `KS`, `Coef_KS`, `Coef_KS_ref` | Distância máxima entre as distribuições acumuladas do alvo e da previsão, e quanto ela melhora a de uma normal com a mesma média e o mesmo desvio-padrão |

**`AletricasDistribuicaoProbabilidade(y, matriz_y_prob, discretizador, y_ref=None, y2_ref=None)`**
- Avalia um modelo que prevê uma distribuição para um alvo contínuo: uma probabilidade por faixa do alvo, com as faixas de `discretizador`, um `DiscretizaAlvoRegressivo`.
- As métricas comparam distribuições acumuladas (`KS`, `Coef_KS`).

## `AleMLAnalysis`: avaliação de modelos

**`AvaliaClassificacao(df, col_alvo, col_prob, ...)`**
- Junta as métricas de `AletricasClassificacao` com a análise por segmento:
  - `calcula_metricas_condicionais(df, col_ref=[...])` divide cada variável em faixas (`num_div`, `num_cat`) e recalcula as métricas em cada uma;
  - `grafico_probabilidade_condicional(col)` mostra, por faixa, a taxa real do alvo contra a probabilidade média, que é uma calibração por segmento;
  - `grafico_metricas_condicionais(col, metricas=[...])` mostra, por faixa, qualquer métrica da tabela acima;
  - `barra_calor(col)` desenha, numa barra de cores, a taxa do alvo em cada faixa da variável;
  - `ganho_info()` e `razao_ganho_info()` medem quanto cada variável informa sobre o alvo.

**`AvaliaDatasetsClassificacao(dict_dfs, col_alvo, col_prob, num_div_prob=None, num_div=None, num_cat=None, unit=None, chave_treino="Treino")`**
- Faz o mesmo em várias bases, como `{"Treino": ..., "Teste": ...}`, com tabelas e gráficos lado a lado.
- A base `chave_treino` é a referência: dela saem os cortes de decisão aplicados às outras bases e a taxa usada nas métricas `_ref`.
- Nas métricas por faixa, as faixas de cada variável são calculadas sobre a união das bases, para serem as mesmas em todas.
- Métodos:
  - `valor_metricas()` e `valor_metricas_condicionais(col)`, que devolve um dicionário com uma tabela por base;
  - `grafico_roc()`, `grafico_ks()`, `grafico_informacao()`, `grafico_distribuicao(col)`, `grafico_probabilidade_condicional(col)` e `grafico_metricas_condicionais(col, metricas)`.

**`AvaliaRegressao`, `AvaliaDatasetsRegressao`, `AvaliaDistribuicaoProbabilidade`, `AvaliaDatasetsDistribuicaoProbabilidade`**
- São as versões para regressão e para modelos de distribuição, com os mesmos métodos.
- Na regressão, `grafico_media_condicional(col)` mostra a média do alvo e a da previsão por faixa.
- Na distribuição, `grafico_distribuicao_condicional(col)` faz o mesmo para a distribuição prevista.

## `AleLearners`: modelos próprios

**`SeriesMultiGradientClassifier()` e `SeriesMultiGradientRegressor()`**
- São modelos lineares sobre uma expansão em série das variáveis, com regressão logística no classificador.
- O treino é feito em três chamadas:

```python
clf = lrn.SeriesMultiGradientClassifier()
clf.setup_datasets(dict_Xy, laurent=True, ordem=2, apenas_interacoes=False)   # monta a expansão e normaliza
clf.setup_optimizer(alpha=0.001, epsilon=1e-8, early_stop=0, filtro_atualizacao_thetas=True)
clf.fit(num_passos=1000, verbose=1, plot=False)
probs = clf.predict_proba(X)[:, 1]
```

- **`dict_Xy`** é um dicionário `{"nome": (X, y), ...}` com as mesmas colunas, normalmente as divisões de treino de uma validação cruzada.
- **Expansão.**
  - Cada termo é uma potência ou um produto de variáveis até a `ordem` (Taylor). Com `laurent=True`, entram também potências negativas (Laurent).
  - Variáveis binárias não ganham potências.
  - Termos sem variância em algum conjunto saem.
  - Todos os termos são padronizados pela média e pelo desvio-padrão dos conjuntos.
- **Passo.**
  - Em cada passo, o gradiente é calculado em cada conjunto.
  - Com `filtro_atualizacao_thetas=True`, só se movem os parâmetros cujo gradiente tem o mesmo sinal em todos os conjuntos. É a mesma ideia da AND-mask de Parascandolo et al. (2021).
  - O tamanho do passo é normalizado pela norma do gradiente, com `alpha` como teto, e diminui perto da convergência.
- **Parada.**
  - O treino para por número de passos, por melhora menor que `epsilon` ou por `early_stop` passos sem ganho (0 desliga a parada antecipada).
  - O modelo guarda os melhores parâmetros de cada conjunto. O modelo final é a média deles, sem os termos de coeficiente nulo.
- **Gráficos.**
  - `grafico_vies_variancia()` mostra a métrica de cada conjunto ao longo dos passos: R² na regressão, coeficiente de LogLoss na classificação.
  - `grafico_importancias()` mostra a importância por variável, e `grafico_importancias_termos(num_vars)` por termo da expansão.

**`DiscretizaAlvoRegressivo(y, num_div=10)`**
- Divide um alvo contínuo em faixas quase uniformes, para treinar um classificador por faixa e estimar a distribuição do alvo, como em [`exemplos/03_house_prices_distribuicao.ipynb`](../exemplos/03_house_prices_distribuicao.ipynb).
- `calcula_probabilidades(y)` devolve a distribuição observada.
- `grafico_distribuicao_probabilidade(probs)` desenha a distribuição prevista.

**`TaylorLaurentExpansion` e `ChecaMultiDatasets`**: uso interno, para os termos da expansão e para validar `dict_Xy`.

## `AleExplainers`: explicabilidade

Servem para qualquer modelo com `predict_proba`, na classificação, ou `predict`, na regressão, inclusive os do scikit-learn e o CatBoost.

**`ImportanciaVariaveisClassificacao(clf, df, col_alvo, cols_features=None, num_loop=5, random_state=None)`**
- Importância por permutação: cada variável é embaralhada `num_loop` vezes, e mede-se quanto o LogLoss piora.
- Métodos:
  - `retorna_piora_logloss()` e `retorna_incertezas_piora()` devolvem a piora média e o desvio dela;
  - `retorna_importancias()` e `retorna_incertezas()` devolvem os mesmos valores normalizados para somar 1.
- `ImportanciaVariaveisRegressao` faz o mesmo com o MSE.

**`CamClassificacao(clf, df, df_exp, cols_features=None, num_loop=5, random_state=None)`**
- Explicação local por perturbação: para cada exemplo de `df_exp` e cada variável, o valor é trocado `num_loop` vezes por valores sorteados de `df`, e mede-se quanto a probabilidade muda.
- `styler_mapa_calor(df_exp, ordenar_linhas=False, ordenar_colunas=False, opacity_text=False)` devolve um `Styler` do pandas, com uma cor por célula:
  - laranja quando o valor real da variável aumenta a probabilidade daquele exemplo;
  - azul quando diminui.
- `CamRegressao` faz o mesmo para a previsão de um regressor.

## `AleCFunctions` e `AleCFunctionsLearners`

Funções numéricas usadas pelas classes acima e compiladas com `numba.jit_module(nopython=True)`:
- contagem de valores distintos com o alvo;
- a busca das fronteiras de discretização;
- entropia e ganho de informação;
- distribuições acumuladas, KS e Kendall tau;
- os passos do gradiente dos modelos.

A primeira chamada de cada uma é mais lenta, por causa da compilação.

## Comportamentos a conhecer

- **Versões.**
  - O código usa `np.str`, removido no numpy 1.24, e `OneHotEncoder(sparse=...)`, obsoleto desde o scikit-learn 1.2 e removido no 1.4. Por isso [`requirements.txt`](../requirements.txt) fixa as versões testadas.
  - A única mudança feita em 2026 é a das funções `unicos_qtds_alvos` e `unicos_qtds_alvos2`, que passaram a acumular o alvo como número decimal. Sem ela, o numba a partir da versão 0.50 recusa alvos decimais.
- **Rótulos das faixas.** A coluna `Str` usa o menor número de algarismos significativos que mantém os cortes distintos, então é uma aproximação: a faixa de 30,75 a 36,25 pode aparecer como `[30, 40)`. Os limites exatos estão em `Min` e `Max`.
- **`SeriesMultiGradientRegressor`.** O alvo é escalado pela raiz da média, e não pelo desvio-padrão (`AleLearners.py`, em `setup_datasets`). Para alvos positivos isso só muda a escala do passo. Se a média do alvo for negativa, o treino dá erro.
- **`CortaIntervalosR2(balancear=True)`.** A soma dos quadrados do alvo por valor distinto omite o último valor (`unicos_qtds_alvos2` não acrescenta o último acumulado), o que distorce um pouco a última faixa. Com `balancear=False`, o padrão, isso não acontece.
- **Sementes.**
  - Em `ImportanciaVariaveis*`, qualquer `random_state` diferente de `None` fixa a semente em 42.
  - O Kendall tau e as permutações dependem de sorteio, então variam um pouco entre execuções, como mostram os notebooks rodados em 2026.

# Facebook Comments Volume

Trata-se de um problema de regressão descrito no
artigo [Facebook Comments
Volume](http://uksim.info/uksim2015/data/8713a015.pdf).

O objetivo que se
deseja alcançar nesta solução é de acordo com as informações de entrada sobre um
determinado post, predizer qual o número de comentários ele irá receber em uma
determinada quantidade de horas.

Os dados foram obtidos através de um
***crawler*** nas páginas do facebook, logo após foram pré-processadas e
divididas temporalmente. O [conjunto de dados](./Dataset.zip) fornecidos pelo
problema estão separados em dados para treino e para teste.

De acordo com o
autor, foram obtidos 5 diferentes tipos de dados para o treino, separados
temporalmente em Variante 1, 2, 3, 4 e 5.

## Dados

Há um total de 54
***features***, essas features são representações para as colunas abaixo.
|Número da coluna|Nome|Origem da feature|Descrição|
|----------------|----|-----------------|---------|
|1|Popularidade/*Likes* da
páginaPage|*Feature* da página|Define a popularidade da página ou o suporte para
a fonte dos documentos.|
|2|Página Checking|*Feature* da página|Descreve quantos
individuos diferentes visitaram a página|
|3|Falano sobre a página|*Feature* da
página|Define o interese diário de indivídos em relação ao conteúdo da página.
Ou seja, pessoas que voltam ou tem alguma ação sobre a página.|
|4|Categoria da
página|*Feature* da página|Define a categoria da página. Ex.: lugar,
instituição...|
|5 - 29|Derivado|*Feature* derivada |São features que foram
associadas a página calculando através de outras features básicas.|
|30|CC1|*Feature* essencial|Número total de comentários antes do tempo base
selecionado.|
|31|CC2|*Feature* essencial|O número de comentários nas 24 horas
atrás|
|32|CC3|*Feature* essencial|O número de comentários nas 48
horas atras e 24 horas relativas ao tempo base.|
|33|CC4|*Feature* essencial|O
número de comentários nas primeiras 24 horas após a publicação, mas antes do
tempo base.|
|34|CC5|*Feature* essencial|Diferença entre CC2 e CC3.|
|35|Tempo
base|Outra *feature*|Tempo selecionado para simular o cenário.|
|36|Tamanho do
*post*|Outra *feature*|Contagem de caracteres no *post*.|
|37|*Post*
compartilhados|Outra *feature*|É a contagem de pessoas que compartilharam o
*post* em sua linha do tempo.|
|38|Promoção do estado do *Post*|Outra
*feature*|Pessoas que promoveram (1) ou não (0) a página que apareceu na lista
de novas notícias.|
|39|H Local|Outra *feature*|Descreve o número de horas que
passaram para o alvo de comentários recebidos.|
|40-46|Dia da semana da
publicação|Dia da semana|Representa o dia da semana que foi publicado o post
(Domingo até Sábado).|
|47-53|Dia do tempo base|Dia da semana|Representa o dia
(Domingo até Sábado) que foi selecionado o tempo base.|
|54|Variável alvo|Alvo|O
número de comentários nas próximas H horas.|

## Obtenção dos dados

Os dados
estão organizados na pasta
`./Dataset/Training/Features_Variant_X.csv`, em que X
é um dos valores de
variação (1, 2, 3, 4, 5).

|Número de registros|Nome do
arquivo|Tamanho|
|-------------------|---------------|-------|
|40949|Features_Variant_1.csv|13M|
|81312|Features_Variant_2.csv|26M|
|121098|Features_Variant_3.csv|39M|
|160424|Features_Variant_4.csv|51M|
|199030|Features_Variant_5.csv|64M|


A primeira amostra que trabalharemos
consistem em 40948 inputs de treinamento e 10043 inputs de teste, ambos com 53
colunas cada.
Lembrando que para rodar o código parser.py é necessário
descompactar o arquivo do dataset.
Primeiro precisamos obter as pastas onde se
encontram os arquivos. Para isso usaremos o import os (ou Operating system) e
como a extenção dos arquivos é .csv, será utilizada a biblioteca
[pandas](http://pandas.pydata.org/).

```python
import os
# Load dirs name
cur_dir = os.path.realpath('.')
data_dir = os.path.join(cur_dir,'Dataset')
# Obtaining directories
train_dir = os.path.join(data_dir,'Training')
print(train_dir)
test_dir = os.path.join(data_dir,'Testing', 'TestSet')
print(test_dir)
```

#### Obtendo local dos arquivos

Agora com os diretórios em mãos podemos obter
os arquivos do tipo csv para teste e treinamento. Ao final iremos ordenar a
lista para podemos acessar o arquivos que queremos pelo indice, ja que não
necessáriamente ele vai ler em ordem alfabetica.

```python
list_train = []
list_test = []
# Obtain train files
for x in os.listdir(train_dir):
    if(x.endswith(".csv")):
        list_train.append(os.path.join(train_dir, x))

for x in os.listdir(test_dir):
    if(x.endswith(".csv")):
        list_test.append(os.path.join(test_dir, x))

# Sorting array to access the required ones
list_train = sorted(list_train)
list_test = sorted(list_test)

print("Train List: ", list_train)
print("Test List: ", list_test)
```

Tendo os paths dos arquivos em mãos podemos obter os dados. Para fazer isso
de
forma eficiente podemos utilizar a biblioteca Pandas.

#### Definição das colunas

Para a correa montagem do data frame, definimos as
colunas e o
caminho para o arquivo onde está presente o primeiro dataset do
problema.
Portanto, é esperado que, caso o pandas leia corretamente, tenham
40949
registros em 54 colunas.

```python
columns = ["Page Popularity/likes", "Page Checkinsâ€™s", "Page talking about",
           "Page Category", "Derived", "Derived", "Derived", "Derived",
           "Derived", "Derived", "Derived", "Derived", "Derived",
           "Derived", "Derived", "Derived", "Derived", "Derived",
           "Derived", "Derived", "Derived", "Derived", "Derived",
           "Derived", "Derived", "Derived", "Derived", "Derived",
           "Derived", "CC1", "CC2", "CC3", "CC4", "CC5", "Base time",
           "Post length", "Post Share Count", "Post Promotion Status", "H Local",
           "Post Sunday", "Post Monday", "Post Tuesday", "Post Wednesday", "Post Thursday", "Post Friday", "Post Saturday",
           "Base Sunday", "Base Monday", "Base Tuesday", "Base Wednesday", "Base Thursday", "Base Friday", "Base Saturday",
           "Target Variable"]
print(len(columns), columns)
```

#### Leitura dos dados

Agora, faremos a leitura dos dados com o suporte da
biblioteca pandas, em que passamos o local do arquivo e o nome das colunas.

```python
import pandas
trainData = pandas.read_csv(list_train[0], names=columns)
testData = pandas.read_csv(list_test[0], names=columns)
print('Train file: ', list_train[0])
print('Test file: ', list_test[0])
print("Quantidade de dados de treinamento")
print(len(trainData))
print("Quantidade de dados de teste")
print(len(testData))
trainData.head()
```

## Tratamento

Será realizada as etapas de *feature selection* e *feature
engineering*.

#### Correlação entre features

Será realizada uma análise da
correlação entre as *features*. Visto que há um total de 24 colunas que foram
fruto de engenharia de caracerísticas, e que, o autor não especificou quais
foram as operações realizadas entre elas e portanto, esta análize ajudará a
identificar as relações entre as *features*.

##### O que é

A correlação entre duas variáveis é quando existe algum laço matemático que
envolve o valor de duas variáveis de alguma forma ([ESTATÍSTICA II - CORRELAÇÃO
E REGRESSÃO](http://www.ctec.ufal.br/professor/mgn/05CorrelacaoERegressao.pdf)).

Uma das maneiras mais simples de se identificar a correlação entre duas
variáveis é plotando-as em um gráfico, para tentar identificar alguma relação
entre elas.

Suponha os seguintes dados: X representa o número de visitas totais em uma
página do facebook e Y o número de curtidas que esta página possui.

|Visitas|Curtidas|
|-------|--------|
|25000|5000|
|1000|95|
|10000|1500|
|12000|1900|
|20005|3700|
|5000|1200|
|3000|600|
|15000|3000|

```python
%matplotlib inline
X, Y = 0, 1

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

fig, (ax, ax2) = plt.subplots(1, 2)
data = np.array([[25000,1000,10000,12000,20005, 5000, 3000, 15000, 250000, 100000 ], [5000,95,1500,1900,3700, 1200, 600, 3000, 50000, 20000]])

ax.plot(data[X], data[Y], '.')

# Calcula regressão linear para a*m + b
m, b, R, p, SEm = linregress(data[X], data[Y])
m,b,R,p,SEm
x2 = np.array([0, data[X].max()])
ax.plot(x2, m * x2 + b)
ax.set_title('linear')

# Dato não linear, relação de raiz quadrada
ax2.set_title('não-linear')
data[Y] = data[X]**(1/6)
chart = ax2.plot(data[X], data[Y], 'o')

```

Com o gráfico, fica bastante claro a correlação das variáveis X e Y do exemplo.
E inclusive o tipo de correlação não-linear.

#### Tipos de correlação

Existem vários métodos para calculo do coeficiente de correlação entre duas
variáveis, pearson, kendall e spearman.
* [Pearson](https://pt.wikipedia.org/wiki/Coeficiente_de_correla%C3%A7%C3%A3o_de
_Pearson): mede o grau da correlação (e a direcção dessa correlação - se
positiva ou negativa) entre duas variáveis de escala métrica (intervalar ou de
rácio/razão).
* [Kendall](https://pt.wikipedia.org/wiki/Coeficiente_de_correla%C3%A7%C3%A3o_ta
u_de_Kendall): medir a correlação de postos entre duas quantidades medidas.
* [Spearman](): A correlação de Spearman entre duas variáveis é igual à
correlação de Pearson entre os valores de postos daquelas duas variáveis.
Enquanto a correlação de Pearson avalia relações lineares, a correlação de
Spearman avalia relações monótonas, sejam elas lineares ou não.

Para nosso problema utilizaremos o método de pearson, pois queremos medir apenas
o grau de correlação entre as variáveis do problema.

```python
import numpy as np
a=trainData.corr('pearson')
a.head()
```

##### Triangulo superior

Como a matriz de correlação gerada pelo dataframe é
uma matriz espelho. Então será removido a parte inferior da matriz

```python
a = a.abs()
np.fill_diagonal(a.values,np.NaN)
upper_matrix = np.triu(np.ones(a.shape)).astype(np.bool)

a=a.where(upper_matrix)
a.head()
```

##### Apenas valores válidos

A matriz de correlação acima nos trás todos os
valores de relacionamento entre cada uma das colunas, logo, desejamos saber
apenas quais são as colunas que possuem uma forte ligação. Usaremos a correlação
forte como sendo > X

```python
a=a.where(a>0.95)
a=a.dropna(how='all', axis=(0,1))
b=a[a.notnull()].stack().index
for c in b:
    print(c, a[c[1]][c[0]])

```

```python
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import seaborn as sns

corr = testData.corr()
sns.set(style="white")
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},
            yticklabels=columns,xticklabels=columns)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()
```

### Validação

Após identificar quais features possuem uma forte correlação com outras,
precisamos validar se removendo as colunas com forte correlação irão influênciar
na precisão do modelo proposto.

#### Cross-Validation

A validação cruzada é uma dos modelos para validar se um modelo está tendo uma
alta acertividade, separando um subconjunto dos dados disponíveis para realizar
a validação, e o seu complemento para o treino ([TÉCNICAS DE APRENDIZAGEM DE
MÁQUINA PARA PREVISÃO DE SUCESSO EM
IMPLANTESDENTÁRIOS](http://tcc.ecomp.poli.br/CarolinaBaldisserotto.pdf)). Este
tipo de validação é bom para detectar [*overffiting* e
*underffiting*](http://docs.aws.amazon.com/machine-learning/latest/dg/model-fit-
underfitting-vs-overfitting.html) e possibilita generalizar a solução
adequadamente para o domínio.

Um dos métodos mais comuns para aplicar a validação cruzada é o k-fold, neste
método, separa-se o conjunto de dados em **k** subconjuntos, utiliza-se
**(k-1)** subconjuntos para treinar o modelo e 1 para validálo. Este processo é
repetido **k** vezes sempre excluindo 1 subconjunto diferente para cada iteração
([Cross-Validation](http://docs.aws.amazon.com/machine-learning/latest/dg/cross-
validation.html)).

#### R²

O [R²](http://leg.ufpr.br/~silvia/CE003/node76.html) é chamado de coeficiente de
determinação. Ele é uma variável que é explicada pela variabilidade de outras
variáveis, conhecido como quadrado do coeficiente de correlação de Pearson, ou
seja, ele indica o quanto da variação total está relacionada aos valores
analisados em pares.

Os valores de R² variam entre -infinito e 1, pois são determinados pela fórmula:

\begin{equation*}
R^2 = \left( 1 - \frac{(Variação explicada de Y)}{(Variação total de Y)}\right)
\end{equation*}

É possível que a variação total seja próxima e 0 e a variação explicada de Y
seja grande, fazendo com que gere valores negativos.
\begin{equation*}
Variação total de Y  =  \sum_{k=0}^n (Y_i - \bar{Y})^2
\end{equation*}
\begin{equation*}
Variação total de Y  =  \sum_{k=0}^n (Y_i - F_i)^2
\end{equation*}

```python
%%time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor


y_test, x_test = testData.loc[:, 'Target Variable'], testData.drop('Target Variable', 1)
Y, X = trainData.loc[:, 'Target Variable'], trainData.drop('Target Variable', 1) 

regressor = DecisionTreeRegressor()

regressor.fit(X, Y)
y = regressor.predict(x_test)

score = cross_val_score(regressor, X, Y, scoring='neg_mean_squared_error')
print(regressor.score(X, Y), regressor.score(x_test, y_test))
```

#### Média quadrática do erro

A [média quadrática do erro](http://www.statisticshowto.com/mean-squared-error/)
é uma métrica para modelos de regressão que indica o quanto os pontos estão se
distânciando da reta traçada. A distância é a medida de **erro** da linha e o
quadrado é útil para obter os valores positivos e amplifica (adiciona maior
peso) para quando o ponto se distância bastante.

Nesta seção, ele será utilizado para determinar o quanto o modelo de regressão
melhorou ou piorou após aplicar a remoção das colunas que possuem forte
correlação.

```python
%%time

filteredData = pandas.read_csv(list_train[2], names=columns)

drop_columns = ['Target Variable', 'Derived.15', 'Derived.16', 'Derived.3', 'Derived.7',
                'Derived.12', 'Derived.17', 'Derived.18', 'Derived.19', 'Derived.24',
                'Derived.21', 'Derived.20', 'CC4']

YY, XX = filteredData.loc[:, 'Target Variable'], filteredData.drop(drop_columns, 1)
y_test, x_test = testData.loc[:, 'Target Variable'], testData.drop(drop_columns, 1)

regressor = DecisionTreeRegressor()

filteredScore = cross_val_score(regressor, XX, YY, scoring='neg_mean_squared_error')
regressor.fit(XX,YY)
yy = regressor.predict(x_test)
print(regressor.score(XX, YY), regressor.score(x_test, y_test))
```

#### Normalização

Normalize samples individually to unit norm.
Each sample (i.e. each row of the data matrix) with at least one non zero
component is rescaled independently of other samples so that its norm (l1 or l2)
equals one.


```python
%%time
from sklearn.preprocessing import Normalizer

XXX, YYY = Normalizer().fit_transform(trainData.drop('Target Variable', 1)), trainData.loc[:, 'Target Variable']
y_test, x_test = testData.loc[:, 'Target Variable'], testData.drop('Target Variable', 1)
regressor = DecisionTreeRegressor()

normalizedScore = cross_val_score(regressor, XXX, YYY, scoring='neg_mean_squared_error')
regressor.fit(XXX, YYY)
yyy = regressor.predict(x_test)
print(regressor.score(XXX, YYY), regressor.score(x_test, y_test))
```

#### Resultados

Obervou-se que com a remoção das colunas dependentes, não houve grandes
melhorias na performance do modelo Regression tree. A seguir é apresentada uma
métrica diferente para esta análise o mean squared error.

```python
main_score = "Score: {} +- {}".format(score.mean(), score.std())
main_filtered = "Score: {} +- {}".format(filteredScore.mean(), filteredScore.std())
main_normalized = "Score: {} +- {}".format(normalizedScore.mean(), normalizedScore.std())

main_score, main_filtered, main_normalized
```

Na plotagem do gráfico a baixo, é visível que não se concretizou uma linha
quando se faz a plotagem em relação ao dado predito e o dado real.

```python
max_value = max(y_test.max(), y.max(), yy.max(), yyy.max())
min_value = min(y_test.min(), y.min(), yy.min(), yyy.min())
plt.scatter(y_test, y, c='green', marker='o', s=10, alpha=0.8, label='Test with Trained Data')
plt.scatter(y_test, yy, c='blue', marker='v', s=10, alpha=0.8, label='Filtered data')
plt.scatter(y_test, yyy, c='red', marker='*', s=10, alpha=0.8, label='Filtered data')
plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='-', linewidth=2)

plt.tight_layout()
plt.show()
```

# Machine Learning

## Facebook Comment Volume

Para o dado problema, foi feito um levantamento dos modelos a serem utilizados
para a proposta de solução. Os modelos selecionados forma baseados na literatura
de artigos, nos quais foram feitos estudos sobre a mesma massa de dados, ou
semelhantes, que está sendo trabalhada neste problema. Com base nos artigos [Kam
aljot](https://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset)
e [Kamaljot, Ranjeet](http://uksim.info/uksim2015/data/8713a015.pdf), ambos
tratam o problema utilizando modelos de regressão supervisionado. Em ambos os
artigos, o modelo no qual foi retratado como melhor desempenho para a solução,
foi o *Decision Tree*.

Os modelos selecionados para testes neste problema foram:
* Decision tree
* Random Forest
* KNM

## Tratamento da Base de dados

Tratamento para a base de testes e treino.

```python
import time
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Set features and independent variables vector
X_train = trainData.iloc[:, :-1].values
y_train = trainData.iloc[:, -1].values

X_test = testData.iloc[:, :-1].values
y_test = testData.iloc[:, -1].values

print("X values and Y values ready for training and testing!!!")
```

```python
def plot_graphs(y_train, y_train_pred, y_test, y_test_pred):
    xy_min = 0
    xy_max = 1500

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 5))
    
    ax1.scatter(y_train, y_train_pred, c='blue', marker='o', s=50, alpha=0.7, label='Test with Trained Data')
    ax1.set_title('Prediction on Train set')
    ax1.set_xlim([xy_min, xy_max])
    ax1.set_ylim([xy_min, xy_max])
    ax1.plot([xy_min, xy_max], [xy_min, xy_max], color='red', linestyle='-', linewidth=2)
    
    ax2.scatter(y_test, y_test_pred, c='darkorange', marker='o', s=50, alpha=0.8, label='Test with Test Data')
    ax2.set_title('Prediction on Test set')
    ax2.set_xlim([xy_min, xy_max])
    ax2.set_ylim([xy_min, xy_max])
    ax2.plot([xy_min, xy_max], [xy_min, xy_max], color='red', linestyle='-', linewidth=2)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(15, 5))
    plt.scatter(y_train_pred, y_train - y_train_pred, c='black', marker='o', s=75, alpha=0.7, label='Training data')
    plt.scatter(y_test_pred, y_test - y_test_pred, c='lightgreen', marker='s', s=75, alpha=0.7, label='Test data')
    plt.xlabel('Predicted values', fontsize=18)
    plt.ylabel('Residuals', fontsize=18)
    plt.legend(loc='upper right')
    plt.plot([0, 1250],[0, 0], color='red', linestyle='-', linewidth=2)
    plt.xlim([-20, 1250])
    plt.ylim([-1000, 1500])
    plt.tight_layout()
    plt.show()
```

## Decision Tree Regression

Uma árvore de regressão é idêntica a uma árvore de decisão porque também é
formada por um conjunto de nós de decisão, perguntas, mas o resultado, em vez de
uma categoria, é um escalar. As mesmas são boas candidatas, pois elas capturam
iterações complexas nos dados.

Para o estudo foi feito os testes do modelo utilizando oa parêmetros *default*
da árvore. sendo esse encontrardo neste [link](http://scikit-learn.org/stable/mo
dules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTre
eRegressor)

```python
from sklearn.tree import DecisionTreeRegressor

def decision_tree_regressor(X_train, y_train, X_test, y_test):
    t0 = time.time()
    print("Runnning Regression Decision Tree...")

    regressor = DecisionTreeRegressor(max_depth=100)
    regressor.fit(X_train, y_train)
    
    y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)
    
    plot_graphs(y_train, y_train_pred, y_test, y_test_pred)
    
    print("R² Score, on Training set: %.3f, on Testing set: %.3f" % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))
    print("Mean Squared Error Score on Testing set: %.2f" % (mean_squared_error(y_test, y_test_pred)))
    print("It took %.2f" % (time.time() - t0), "seconds to run Decision Tree Regression")
    
decision_tree_regressor(X_train, y_train, X_test, y_test)
```

## Random Forest Regression

É um meta-estimador que se adapta a uma série de árvores de decisão de
classificação em várias sub-amostras do conjunto de dados e utiliza a média para
melhorar a precisão preditiva e controlar a sobreposição.

O modelo tem como parâmetro livre a seleção da quantidade de árvores de decisão.

```python
from sklearn.ensemble import RandomForestRegressor

def random_forest_regressor(X_train, y_train, X_test, y_test):
    t0 = time.time()
    n_trees = 20
    print("Runnning Random Forest with",n_trees,"Trees...")
    
    regressor = RandomForestRegressor(n_estimators=n_trees, random_state=1, n_jobs=-1)
    regressor.fit(X_train, y_train)
    
    y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)   
    
    plot_graphs(y_train, y_train_pred, y_test, y_test_pred)
    
    print("R² Score, on Training set: %.3f, on Testing set: %.3f" % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))
    print("Mean Squared Error Score on Testing set: %.2f" % (mean_squared_error(y_test, y_test_pred)))
    print("It took %.2f" % (time.time() - t0), "seconds to run Random Forest Regression with", n_trees, "trees")
    
random_forest_regressor(X_train, y_train, X_test, y_test)
```

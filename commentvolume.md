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

```{.python .input  n=1}
import os
# Load dirs name
cur_dir = os.path.realpath('.')
data_dir = os.path.join(cur_dir,'Dataset')
# Obtaining directories
train_dir = os.path.join(data_dir,'Training')
print(train_dir)
test_dir = os.path.join(data_dir,'Testing')
print(test_dir)
```

#### Obtendo local dos arquivos

Agora com os diretórios em mãos podemos obter
os arquivos do tipo csv para teste e treinamento. Ao final iremos ordenar a
lista para podemos acessar o arquivos que queremos pelo indice, ja que não
necessáriamente ele vai ler em ordem alfabetica.

```{.python .input  n=2}
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

```{.python .input  n=3}
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

```{.python .input  n=53}
import pandas
trainData = pandas.read_csv(list_train[0], names=columns)
testData = pandas.read_csv(list_test[0], names=columns)
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

```{.python .input  n=54}
%matplotlib inline
X, Y = 0, 1

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

fig, (ax, ax2) = plt.subplots(1, 2)
data = np.array([[25000,1000,10000,12000,20005, 5000, 3000, 15000], [5000,95,1500,1900,3700, 1200, 600, 3000]])

ax.plot(data[X], data[Y], '.')

# Calcula regressão linear para a*m + b
m, b, R, p, SEm = linregress(data[X], data[Y])
m,b,R,p,SEm
x2 = np.array([0, data[X].max()])
ax.plot(x2, m * x2 + b)
ax.set_title('linear')

# Dato não linear, relação de raiz quadrada
ax2.set_title('não-linear')
data[Y] = data[X]**(1/2)
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

```{.python .input  n=55}
import numpy as np
a=trainData.corr('pearson')
a
```

##### Triangulo superior

Como a matriz de correlação gerada pelo dataframe é
uma matriz espelho. Então será removido a parte inferior da matriz

```{.python .input  n=56}
a = a.abs()
np.fill_diagonal(a.values,np.NaN)
upper_matrix = np.triu(np.ones(a.shape)).astype(np.bool)

a=a.where(upper_matrix)
a
```

##### Apenas valores válidos

A matriz de correlação acima nos trás todos os
valores de relacionamento entre cada uma das colunas, logo, desejamos saber
apenas quais são as colunas que possuem uma forte ligação. Usaremos a correlação
forte como sendo > X

```{.python .input  n=83}
a=a.where(a>0.95)
a=a.dropna(how='all', axis=(0,1))
b=a[a.notnull()].stack().index
for c in b:
    print(c, a[c[1]][c[0]])

```

```{.python .input  n=58}
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

#### Média quadrática do erro

A [média quadrática do erro](http://www.statisticshowto.com/mean-squared-error/)
é uma métrica para modelos de regressão que indica o quanto os pontos estão se
distânciando da reta traçada. A distância é a medida de **erro** da linha e o
quadrado é útil para obter os valores positivos e amplifica (adiciona maior
peso) para quando o ponto se distância bastante.

Nesta seção, ele será utilizado para determinar o quanto o modelo de regressão
melhorou ou piorou após aplicar a remoção das colunas que possuem forte
correlação.

```{.python .input  n=112}
%%time
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()

y_test, x_test = testData.loc[:, 'Target Variable'], testData.drop('Target Variable', 1)
Y, X = trainData.loc[:, 'Target Variable'], trainData.drop('Target Variable', 1) 

regressor.fit(X, Y)
y = regressor.predict(x_test)

score = cross_val_score(regressor, X, Y, scoring='neg_mean_squared_error')
```

```{.python .input  n=120}
%%time
filteredData = pandas.read_csv(list_train[2], names=columns)


drop_columns = ['Target Variable', 'Derived.15', 'Derived.16', 'Derived.3', 'Derived.7',
                'Derived.12', 'Derived.17', 'Derived.18', 'Derived.19', 'Derived.24',
                'Derived.21', 'Derived.20', 'CC4']

YY, XX = filteredData.loc[:, 'Target Variable'], filteredData.drop(drop_columns, 1)
y_test, x_test = testData.loc[:, 'Target Variable'], testData.drop(drop_columns, 1)

filteredScore = cross_val_score(regressor, XX, YY, scoring='neg_mean_squared_error')
regressor.fit(XX,YY)
yy = regressor.predict(x_test)
```

```{.python .input  n=121}
main_score = "Score: {} +- {}".format(score.mean(), score.std())
main_filtered = "Score: {} +- {}".format(filteredScore.mean(), filteredScore.std())
main_score, main_filtered
```

```{.python .input  n=127}
regressor.score(x_test,y_test)
```

```{.python .input  n=122}
plt.scatter(y_test, y, c='green', marker='o', s=10, alpha=0.5, label='Test with Trained Data')
plt.scatter(y_test, yy, c='blue', marker='v', s=10, alpha=0.5, label='Filtered data')
plt.tight_layout()
plt.show()
```

# Machine Learning

## Facebook Comment Volume

## Regression

```{.python .input  n=63}
import time

# Set features and independent variables vector
X = trainData.iloc[:, :-1].values
y = trainData.iloc[:, -1].values

print("X values and Y values ready for training!!!")
```

## Decision Tree Regression
![](http://scikit-
learn.org/stable/_images/sphx_glr_plot_tree_regression_001.png)

```{.python .input  n=64}
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import explained_variance_score

def decision_tree_regressor(X, y):
    print("Runnning Regression Decision Tree...")
    
    t0 = time.time()
    
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X, y)

    y_predicted = regressor.predict(X)

    print("Label: ", y)
    print("Predicted: ", y_predicted)

    print("Decision Tree Overall Accuracy: {0:.4f}".format(explained_variance_score(y, y_predicted)), "%")
    
    print("It took {0:.2f}".format(time.time() - t0),"seconds to run Decision Tree Regressor") 
    
decision_tree_regressor(X, y)
```

## Random Forest Regression

```{.python .input  n=65}
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score

def random_forest_regressor(X,y):
    t0 = time.time()
    # n_trees = number of trees
    n_trees = 10
        
    print("Runnning Random Forest with",n_trees,"Trees...")
    regressor = RandomForestRegressor(n_estimators = n_trees, random_state = 0)
    regressor.fit(X, y)
    
    y_predicted = regressor.predict(X)

    print("Label: ", y)
    print("Predicted: ", y_predicted)
    
    print("Random Forest Overall Accuracy: {0:.4f}".format(explained_variance_score(y, y_predicted)), "%")
    print("It took {0:.2f}".format(time.time() - t0),"seconds to run Random Forest Regression") 
    
random_forest_regressor(X, y)
```

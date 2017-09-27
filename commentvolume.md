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
|35|Tempo base|Outra *feature*|Tempo selecionado para simular o cenário.|
|36|Tamanho do *post*|Outra *feature*|Contagem de caracteres no *post*.|
|37|*Post* compartilhados|Outra *feature*|É a contagem de pessoas que
compartilharam o *post* em sua linha do tempo.|
|38|Promoção do estado do
*Post*|Outra *feature*|Pessoas que promoveram (1) ou não (0) a página que
apareceu na lista de novas notícias.|
|39|H Local|Outra *feature*|Descreve o
número de horas que passaram para o alvo de comentários recebidos.|
|40-46|Dia da semana da publicação|Dia da semana|Representa o dia da semana que
foi publicado o post (Domingo até Sábado).|
|47-53|Dia do tempo base|Dia da
semana|Representa o dia (Domingo até Sábado) que foi selecionado o tempo base.|
|54|Variável alvo|Alvo|O número de comentários nas próximas H horas.|
## Obtenção dos dados

Os dados estão organizados na pasta
`./Dataset/Training/Features_Variant_X.csv`, em que X é um dos valores de
variação (1, 2, 3, 4, 5).

|Número de registros|Nome do arquivo|Tamanho|
|-------------------|---------------|-------|
|40949|Features_Variant_1.csv|13M|
|81312|Features_Variant_2.csv|26M|
|121098|Features_Variant_3.csv|39M|
|160424|Features_Variant_4.csv|51M|
|199030|Features_Variant_5.csv|64M|

Como a
extenção dos arquivos é .csv, será utilizada a biblioteca
[pandas](http://pandas.pydata.org/) para a leitura dos dados

```{.python .input}
from os import path, listdir

current_dir = path.realpath('.')
data_dir = path.join(current_dir, 'Dataset', 'Training')
listdir(data_dir)
```

### Definição

Para a correa montagem do data frame, definimos as colunas e o
caminho para o arquivo onde está presente o primeiro dataset do problema.
Portanto, é esperado que, caso o pandas leia corretamente, tenham 40949
registros em 54 colunas.

```{.python .input}
import pandas as pd
variant1_dir = path.join(data_dir, 'Features_Variant_1.csv')
print(variant1_dir)
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

### Dataframe

Para fazer a leitura do dataframe, será utilizado as colunas de
cada *feature* e o caminho para onde está o dataset

```{.python .input}
pd.read_csv(variant1_dir, names=columns).head()
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

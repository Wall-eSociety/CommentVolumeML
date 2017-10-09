
# Algoritmo K-nearest neighbors

No reconhecimento de padrões o algoritmo KNN é um método não para-métrico usado para classificação e regressão.
Nos dois casos, o input consiste nos k exemplos de treinamento mais proximos no espaço de amostragem. O output depende se o Knn é usado para classificação ou regressão.

KNN é um tipo de aprendizado baseado em instâncias, onde a função é aproximada apenas localmente e toda a computação é deferida até a classificação.


```python
import pandas as pd
import numpy

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

import math 
%matplotlib inline
import matplotlib.pyplot as plt


```


```python
columns = ["Page Popularity/likes", "PageCheckinsâ€™s", "Pagetalkingabout",
           "Page Category", "Derived", "Derived", "Derived", "Derived",
           "Derived", "Derived", "Derived", "Derived", "Derived",
           "Derived", "Derived", "Derived", "Derived", "Derived", 
           "Derived", "Derived", "Derived", "Derived", "Derived", 
           "Derived", "Derived", "Derived", "Derived", "Derived",
           "Derived", "CC1", "CC2", "CC3", "CC4", "CC5", "Basetime",
           "Postlength", "PostShareCount", "PostPromotionStatus", "HLocal",
           "PostSunday", "PostMonday", "PostTuesday", "PostWednesday", "PostThursday", "PostFriday", "PostSaturday",
           "BaseSunday", "BaseMonday", "BaseTuesday", "BaseWednesday", "BaseThursday", "BaseFriday", "BaseSaturday",
           "TargetVariable"]
```

## Train


```python
data_frame = pd.read_csv('Features_Variant_1.csv',delimiter=',', names=columns )
target = data_frame.TargetVariable
x = data_frame.drop('TargetVariable', 1)
```

## Teste


```python
teste = pd.read_csv('Features_Variant_2.csv',delimiter=',', names=columns )
target2 = teste.TargetVariable
y = teste.drop('TargetVariable', 1)
```


```python
def idealK(x,target,y,target2):
    K = 1
    ab = 0
    vetScore = []
    for i in range(200):
        knn = KNeighborsRegressor(n_neighbors=K)
        knn.fit(x,target)
        score = knn.score(y,target2)
        vetScore.append(score)
        if score > ab:
            ab = score
            d = K
        K += 2
    return d,vetScore
D = numpy.arange(1, 400, 2)
K,score = idealK(x,target,y,target2)
print("ideal K: ",K)
plt.plot(D,score)
```


```python
def regressionKnn(x,target,y,target2):
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(x,target)
    vetPredict = knn.predict(y)   
regressionKnn(x,target,y,target2)
```

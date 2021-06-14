---
title: "ExtraTreeClassifier 직접 구현하기"
tags:
  - Machine Learning
  - Project

categories:
    - Machine Learning

last_modified_at: 20121-06-14

use_math: true

toc: true
toc_sticky: true

---

**Implement ExtraTreeClassifier**

## 기본 설정


```python
# 파이썬 ≥3.5 필수 (파이썬 3.7 추천)
import sys
assert sys.version_info >= (3, 5) 

# 사이킷런 ≥0.20 필수
import sklearn
assert sklearn.__version__ >= "0.20"

# 공통 모듈 임포트
import numpy as np
import os

# 노트북 실행 결과를 동일하게 유지하기 위해
np.random.seed(42)

# 깔끔한 그래프 출력을 위해
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# 그림을 저장할 위치
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

# ExtraTreeClassifier implement

## **설계**

ExtraTreeClassifier 모델 구현에 사용할 데이터는

sklearn의 moons dataset을 사용한다.

ExtraTreeClassifier는 기본적으로
RandomForestClassifier에서 특성의 임계값을 랜덤으로 선택하여
편향을 늘리고 분산을 줄이는 기법이기때문에
구현자체는 RandomForestClassifier와 크게 다르지 않다.

엑스트라트리에서는 가장좋은 임계값을 사용하지 않고, 특성의 임계값을 랜덤으로 지정하여 더빠르게 학습을 진행한다.

이는 랜덤포레스트와 엑스트라트리가 내부적으로 사용하는 모델은 DecisionTreeClassifier인데, 파라미터 splitter의 값을 "best"로 주면 랜덤포레스트가 되고, "random"을 주게되면 엑스트라트리가 되게된다.


```python
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
```

사이킷런의 train_test_split을 이용해 트레인셋과 테스트 셋을 나눠준다.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

엑스트라트리 모델의 구조 설명을 위해 

DecisionTreeClassifier모델을 사용한다. 

투표방식은 소프트방식을 사용하려면

확률을 이용해야 하는데, 이를 지원하는 여러 모델중 하나가 SVC이기 때문이다.

크기가 100인 subset을 사용하는 1000개의 DecisionTree모델을 만든다.


```python
from sklearn.model_selection import ShuffleSplit

n_trees = 1000
n_instances = 100

mini_sets = []

rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances, random_state=42)
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))
```

1000개의 나무(tree)로 숲(forest)을 만든다.

넣어주면 predict_proba를 사용할수 있다.

1000개의 나무들을 subset을 이용하여 훈련을 한 후,

각 나무들의 정확도의 평균을 구한다.

엑스트라트리는 기본적으로 특성과 특성의임계값을 랜덤으로 하기때문에
splitter는 random을 사용한다.

랜덤프로스트는 특성은 랜덤으로, 특성의임계값은 최적을 선택하기때문에
splitter는 best를 선택하였다.


```python
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

forest = [clone(DecisionTreeClassifier(splitter = "random", max_features="auto")) for _ in range(n_trees)]

accuracy_scores = []

for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    clf = make_pipeline(StandardScaler(), tree)
    clf.fit(X_mini_train, y_mini_train)

    y_pred = clf.predict(X_test)

    accuracy_scores.append(accuracy_score(y_test, y_pred))

np.mean(accuracy_scores)
```




    0.7794770000000001



기존 코드에서는 mode를 사용해 가장 많이 뽑힌 클래스를 선택하는, 직접투표 방식의 

분류를 사용하였는데, 엑스트라트리 모델은 기본적으로 DecisionTreeClassifier를 

사용하고, 이 모델은 확률값(predict_proba)를 보여주기 때문에 간접투표 방식이 더 

좋을거라고 판단을 하였다.

DecisionTreeClassifier 모델은 확률을 알수있는 기능을 제공한다.

이제 테스트셋에 대해 예측을하고, 각 클래스에 속할 확률을 모든 나무에 대해서

list자료형을 이용해 저장한다.


```python
pred_list = []

for tree_index, tree in enumerate(forest):
    pred_list.append(tree.predict_proba(X_test))
```


```python
len(pred_list)
```




    1000



이제 모든 나무들의 데이터에 대한 확률 평균을 구해야한다.

예를들어 33번째 데이터에 대한 확률 정보는 1000개가 있는데, 

1000개의 확률 정보의 평균을 구해, 가장 확률이 높았던 클래스를 선택해

분류하는 방식이다.


```python
class_proba = np.mean(pred_list, axis=0)
```


```python
class_proba
```




    array([[0.177, 0.823],
           [0.08 , 0.92 ],
           [0.787, 0.213],
           ...,
           [0.797, 0.203],
           [0.885, 0.115],
           [0.657, 0.343]])




```python
len(class_proba)
```




    2000



테스트셋의 크기는 2000이였는데, 무사히 모든 테스트셋에 대하여 확률이 구해진것을 알수있다.

이제 남은것은, 가장 확률이 높았던 클래스를 선택하여 분류를 마치는것이다..


```python
y_pred = []
for i in class_proba:
  max=0
  max_index = 0
  for index, j in enumerate(i):
    if j >= max :
      max = j
      max_index = index
  y_pred.append(max_index)
```


```python
accuracy_score(y_pred, y_test)
```




    0.819



분류가 완료되었고, 정확도를 확인해보니 0.84%가 나왔다.

**이제부터 여기까지의 내용을 모두 담아 직접 ExtraTreeClassifier 클래스를 구현한다.**

## **구현**


```python
import random #subset을 섞기위한 random임포트
from sklearn.tree import DecisionTreeClassifier

def make_subset(X, y, ratio):
    """
  subset을 만들어주는 함수이다.
  엑스트라트리 모델은 기본적으로 max_samples의 값이 1.0임을 가정하고
  만들어진 모델이기때문에 그대로 사용한다
   """

    X_subset = []
    y_subset = []
    n_sample = round(len(X) * ratio)
      
    while len(X_subset) < n_sample: #X_subset의 크기가 지정한 크기가 될때까지
        index = random.randrange(len(X)) #index 난수 생성 
        X_subset.append(X[index])
        y_subset.append(y[index])
    return X_subset, y_subset


class MyExtraTreeClassifier:

    def __init__(self,
                 n_estimators=500,
                 max_leaf_nodes=16,
                 random_state=None
                 ):
      
      self.n_estimators = n_estimators
      self.max_leaf_nodes = max_leaf_nodes
      self.random_state = random_state
      self.max_samples = 1.0
      self.trees = []

    def fit(self, X, y):

      for iteration in range(self.n_estimators):
        mini_X_set, mini_y_set = make_subset(X, y, self.max_samples) #모델마다 subset생성하여 다른 subset을 사용하도록 한다.
        #splitter = "random"을통해 특성과 특성의 임계값을 랜덤으로 지정
        self.trees.append(DecisionTreeClassifier(splitter="random", max_features='auto', max_leaf_nodes=self.max_leaf_nodes, random_state=self.random_state).fit(mini_X_set, mini_y_set))
      return self.trees

    def predict(self, data):
      pred_list = []    #각 나무들의 인스턴스에 대한 확률정보가 담겨져있는 리스트
      y_pred = []       #최종 예측값이 저장되는 리스트

      for tree_index, tree in enumerate(self.trees):
          pred_list.append(tree.predict_proba(data))
      mean_list = np.mean(pred_list, axis=0)
      
      for class_prob in mean_list:
        max=0
        max_index = 0

        for index, prob in enumerate(class_prob):
          if prob >= max :
            max = prob
            max_index = index
        y_pred.append(max_index)

      return y_pred
```


```python
import time
```


```python
My_clf = MyExtraTreeClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
t0 = time.time()
models = My_clf.fit(X_train, y_train)
t1 = time.time()
print("MyExtraTreeClassifier took {:.1f}s.".format(t1 - t0))
```

    MyExtraTreeClassifier took 8.6s.
    


```python
y_pred = My_clf.predict(X_test)
```


```python
accuracy_score(y_pred, y_test)
```




    0.854



직접 구현한 ExtraTreeClassifier 모델의 정확도는
약 85.4%가 나왔다.

이제 사이킷런에 내장되어있는 엑스트라 트리 모델과 성능비교를 해보자.


```python
from sklearn.ensemble import ExtraTreesClassifier
t0 = time.time()
extra_clf = ExtraTreesClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
extra_clf.fit(X_train, y_train)
t1 = time.time()
print("ExtraTreeClassifier took {:.1f}s.".format(t1 - t0))
y_pred_extra = extra_clf.predict(X_test)
```

    ExtraTreeClassifier took 0.9s.
    


```python
accuracy_score(y_pred_extra, y_test)
```




    0.8655



아래는 랜덤포레스트 트리로 평가를 한것이다.


```python
from sklearn.ensemble import RandomForestClassifier
t0 = time.time()
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
rnd_clf.fit(X_train, y_train)
t1 = time.time()
print("RandomForestClassifier took {:.1f}s.".format(t1 - t0))
y_pred_rf = rnd_clf.predict(X_test)
```

    RandomForestClassifier took 2.4s.
    


```python
accuracy_score(y_pred_rf, y_test)
```




    0.871



엑스트라트리를 이용하면 분산을 줄어들고 편향은 늘어난것이 확인된다.

편향이 늘면, 그만큼 오차가 있었다는것이고 정확도가 떨어진것을 통해 확인할수 있다.

분산이 줄어든것은, 그만큼 모델복잡도가 줄어들었기때문에 실행시간이 줄어든것을 통해 확인할수 있다.

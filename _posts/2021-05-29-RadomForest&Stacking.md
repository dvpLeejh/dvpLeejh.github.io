---
title: "Implement RandomForestClassifier & Stacking"
tags:
  - Machine Learning
  - Project

categories:
    - Machine Learning

last_modified_at: 20121-05-29

use_math: true

toc: true
toc_sticky: true

---

**implement RandomForestClassifier and StackingClassifier**

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

# RandomForestClassifier 구현

## **설계**

랜덤포레스트 모델 구현에 사용할 데이터는

사이킷런의 moons dataset을 사용한다.


```python
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
```

사이킷런의 train_test_split을 이용해 트레인셋과 테스트 셋을 나눠준다.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

먼저 직접구현한 랜덤포레스트 모델의 구조 설명을 위해 SVC 모델을 사용한다. 

랜덤포레스트 모델의 투표방식은 기본적으로 소프트방식이기 때문에

확률을 이용해야 하는데, 이를 지원하는 여러 모델중 하나가 SVC이기 때문이다.

크기가 100인 subset을 사용하는 1000개의 SVC모델을 만든다.


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

1000개의 나무(SCV())로 숲(forest)을 만든다.

SVC는 기본적으로 확률에 대해서는 알려주지 않는데, 모델에 probability=True값을 

넣어주면 predict_proba를 사용할수 있다.

1000개의 나무들을 subset을 이용하여 훈련을 한 후,

각 나무들의 정확도의 평균을 구한다.


```python
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

forest = [clone(SVC(probability=True)) for _ in range(n_trees)]

accuracy_scores = []

for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    clf = make_pipeline(StandardScaler(), tree)
    clf.fit(X_mini_train, y_mini_train)

    y_pred = clf.predict(X_test)

    accuracy_scores.append(accuracy_score(y_test, y_pred))

np.mean(accuracy_scores)
```




    0.853731



기존 코드에서는 mode를 사용해 가장 많이 뽑힌 클래스를 선택하는, 직접투표 방식의 

분류를 사용하였는데, 랜덤포레스트 모델은 기본적으로 DecisionTreeClassifier를 

사용하고, 이 모델은 확률값(predict_proba)를 보여주기 때문에 간접투표 방식이 더 

좋을거라고 판단을 하였다.

SVC 모델을 설정할때 probability=True 값을 주었으므로 predict_proba를 사용할수 있다.

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




    array([[0.10953949, 0.89046051],
           [0.11436088, 0.88563912],
           [0.8275139 , 0.1724861 ],
           ...,
           [0.77805364, 0.22194636],
           [0.90304155, 0.09695845],
           [0.7641879 , 0.2358121 ]])




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




    0.8425



분류가 완료되었고, 정확도를 확인해보니 0.84%가 나왔다.

**이제부터 여기까지의 내용을 모두 담아 직접 랜덤포레스트모델 클래스를 구현한다.**

## **구현**


```python
import random #subset을 섞기위한 random임포트
from sklearn.tree import DecisionTreeClassifier

def make_subset(X, y, ratio):
    """
  subset을 만들어주는 함수이다.
  랜덤포레스트 모델은 기본적으로 max_samples의 값이 1.0임을 가정하고
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


class MyRandomForestClassifier:

    def __init__(self,
                 n_estimators=500,
                 max_leaf_nodes=16,
                 random_state=None):
      
      self.n_estimators = n_estimators
      self.max_leaf_nodes = max_leaf_nodes
      self.random_state = random_state
      self.max_samples = 1.0
      self.trees = []

    def fit(self, X, y):
      """
      랜덤포레스트 모델은 기본적으로 DecisionTreeClassifier모델을 사용한다.
      """
      for iteration in range(self.n_estimators):
        mini_X_set, mini_y_set = make_subset(X, y, self.max_samples) #모델마다 subset생성하여 다른 subset을 사용하도록 한다.
        self.trees.append(DecisionTreeClassifier(splitter="random", max_leaf_nodes=self.max_leaf_nodes, random_state=self.random_state).fit(mini_X_set, mini_y_set))
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
My_clf = MyRandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)

models = My_clf.fit(X_train, y_train)
```


```python
y_pred = My_clf.predict(X_test)
```


```python
accuracy_score(y_pred, y_test)
```




    0.868



직접 구현한 랜덤포레스트 모델의 정확도는
약 86.7%가 나왔다.

이제 사이킷런에 내장되어있는 랜덤포레스트 모델과 성능비교를 해보자.


```python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)
```


```python
accuracy_score(y_pred_rf, y_test)
```




    0.871



약87.1%로 0.4%정도의 성능차이가 나지만,

이정도면 훌륭하게 랜덤포레스트 모델을 만들수 있었던것 같다.

# **StackingClassifier** **구현**

StackingClassifier 구현에도 사이킷런의 moons_data를 활용한다.


```python
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
```


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

StackingClassifier은 각 모델의 예측값을 훈련데이터로

블렌더가 학습을 하는 분류기이다.

사용할 모델은 직접 구현한 랜덤포레스트모델, SVC 모델이다.


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


random_forest_clf = MyRandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = make_pipeline(StandardScaler(),
                                    SVC(gamma='auto'))
```

각 모델들을 담은 리스트를 만들고, 트레이닝셋에 대해 훈련을 진행해준다.


```python
estimators = [random_forest_clf, svm_clf]
```


```python
from sklearn.tree import DecisionTreeClassifier

def make_prediction(data, estimators):
  """
  StackingClassifier에서 각 모델들이 예측한 결과를 반환해주는 함수이다.
  """
  
  predictions = np.empty((len(data), len(estimators)), dtype=np.float32)

  for index, estimator in enumerate(estimators):
    predictions[:, index] = estimator.predict(data)

  return predictions

class MyStackingClassifier:

  def __init__(self, estimators, final_estimator):
    self.estimators = estimators
    self.final_estimator = final_estimator
  
  def fit(self, X, y):
    # StackingClassifier는 트레이닝 셋이들어오면 또 훈련세트/검증세트로 나누어 훈련을 진행하기에 나눠준다.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    

    for estimator in self.estimators:
      estimator.fit(X_train,y_train)

    #각 모델별로 예측값을 만들어준다.
    predictions = make_prediction(X_val, self.estimators)

    #각 모델들의 예측값을 이용해 블렌더를 만든다. 블렌더에 사용할 모델은 final_estimator를 통해 지정한다.
    blender = self.final_estimator
    blender.fit(predictions, y_val)

  def predict(self, X):
    X_test_predictions = make_prediction(X_test, self.estimators)
    y_pred = self.final_estimator.predict(X_test_predictions)
    return y_pred
```

작동방식은 다음과 같다.

먼저 트레이닝 데이터가 들어오면, 트레이닝 데이터에서 검증데이터를 분리한다.

그뒤 남은 트레이닝 데이터로 각 모델(랜덤포레스트, SVC) 모델을 훈련한뒤

분리해놓은 검증데이터를 통해 각 모델의 예측값을 생성한다.

이렇게 생성된 예측값을 이용해 블렌더, 여기에서는 final_estimator가 학습을 완료한다.



이제 학습을 진행해보자.

블렌더에 사용할 모델은 로지스틱 회귀모델을 사용한다.


```python
from sklearn.linear_model import LogisticRegression

sc = MyStackingClassifier(estimators = estimators, final_estimator = LogisticRegression())
```


```python
sc.fit(X_train, y_train)
```


```python
y_pred = sc.predict(X_test)
```


```python
accuracy_score(y_pred, y_test)
```




    0.873



정확도가 약87.3%가 나왔다.

이제 사이킷런에 내장되어있는 함수와 비교를 해보자.


```python
from sklearn.ensemble import StackingClassifier

estimators = [('rf',RandomForestClassifier(n_estimators=10, random_state=42)),
              ('svr', make_pipeline(StandardScaler(),
                                    SVC(gamma='auto')))]

clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
```


```python
clf.fit(X_train, y_train)
```


```python
y_pred = clf.predict(X_test)
```


```python
accuracy_score(y_pred, y_test)
```

정확도가 87.2%로, 직접구현한 모델보다 0.1%가 낮게나왔다.

스태킹의 개념을 충족시키는 만족할만한 모델을 구현할수 있었던 것같다.

이번 프로젝트에서는 의도적으로 모든 학습에 moons dataset을 사용하였는데
각 직접구현한 모델, 사이킷런 모델의 결과를 살펴보자면

*   직접 구현한 랜덤포레스트 모델 정확도 : 86.7%
*   사이킷런 랜덤포레스트 모델 정확도 : 87.1%
*   직접 구현한 StackingClassifier 모델 정확도 : 87.3%
*   사이킷런 StackingClassifier 모델 정확도 : 87.2%



## 번외편 : 다층 스태킹 분류 모델

 다층 스태킹 모델을 이용해 한번더 성능 측정을 하기로 했다.




```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier

final_layer_rfc = RandomForestClassifier(n_estimators=10, random_state=42)

final_layer_svc = make_pipeline(StandardScaler(), SVC(gamma='auto'))

final_layer = StackingClassifier(
    estimators=[('rf', final_layer_rfc),
                ('svc', final_layer_svc)],
    final_estimator=LogisticRegression()
    )

multi_layer_Classifier = StackingClassifier(
    estimators=[('ridge', RidgeClassifier()),
                ('DCTC', DecisionTreeClassifier(splitter="random", max_leaf_nodes=16, random_state=42)),
                ('neigh', KNeighborsClassifier())],
    final_estimator=final_layer
)
```

모델 구조는 다음과 같다.

---




레이어3                          로지스틱 회귀

레이어2          랜덤포레스트                    소프트백터머신

레이어1          릿지분류기     결정트리 분류기    Neighbors분류기 

데이터가 들어오면 먼저 레이어1의 릿지,결정트리,KNeighbors 분류기에 의해 

예측값이 생성된다. 그 뒤에 레이어2의 랜덤포레스트와 소프트백터머신은 이 

예측값을 가지고 학습을 진행하며, 최종적으로 로지스틱 회귀 모델이

분류를 완료하게된다.


```python
multi_layer_Classifier.fit(X_train, y_train)

y_pred = multi_layer_Classifier.predict(X_test)
```


```python
accuracy_score(y_pred, y_test)
```

결과는 정확도 86%가 나오게되었다.

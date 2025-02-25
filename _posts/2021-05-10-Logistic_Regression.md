---
title: "Logistic Regression 직접 구현하기 및 사진분류"
tags:
  - Machine Learning
  - Logistic Regression
  - Project

categories:
    - Machine Learning

last_modified_at: 20121-05-12

use_math: true

toc: true
toc_sticky: true

---

# 1.Logistic Regression 모델 구현

Logistic Regression 모델을 구현하여 Iris데이터를 분류하고 사이킷런 모델과 성능비교를 할것이다.

**필요한 모듈 임포트**

```python
# 파이썬 ≥3.5 필수
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
CHAPTER_ID = "training_linear_models"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("그림 저장:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
# 어레이 데이터를 csv 파일로 저장하기
def save_data(fileName, arrayName, header=''):
    np.savetxt(fileName, arrayName, delimiter=',', header=header, comments='')
```


**데이터 준비**

붓꽃 데이터셋의 꽃잎 길이(petal length)와 꽃잎 너비(petal width) 특성만 이용한다.

여기에서는 꽃의 품종이  버지니카인지 아닌지 판별할것이다.


```python
from sklearn import datasets
iris = datasets.load_iris()
```


```python
iris.target
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])




```python
X = iris["data"][:, (2, 3)]                  # 꽃잎 길이와 너비
y = (iris["target"] == 2).astype(np.int)
```

0번특성값을 x0이라 판단하기때문에 편향을 추가해야한다.


```python
X_with_bias = np.c_[np.ones([len(X), 1]), X]
```

랜덤 시드 지정(반복해도 같은결과가 나오게 하기위함)


```python
np.random.seed(2042)
```

**데이터셋 분할**

- 훈련 세트: 60%
- 검증 세트: 20%
- 테스트 세트: 20%

의 비율로 데이터셋을 나눠준다.


```python
test_ratio = 0.2                                         # 테스트 세트 비율 = 20%
validation_ratio = 0.2                                   # 검증 세트 비율 = 20%
total_size = len(X_with_bias)                            # 전체 데이터셋 크기

test_size = int(total_size * test_ratio)                 # 테스트 세트 크기: 전체의 20%
validation_size = int(total_size * validation_ratio)     # 검증 세트 크기: 전체의 20%
train_size = total_size - test_size - validation_size    # 훈련 세트 크기: 전체의 60%
```

np.random.permutation() 함수를 이용하여 인덱스를 무작위로 섞는다.


```python
rnd_indices = np.random.permutation(total_size)
```


```python
X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]

X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]

X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]
```


```python
y_train[:5]
```




    array([0, 0, 1, 0, 0])



이제 데이터의 준비가 되었으니
로지스틱 모델을 직접 구현해보자.

**로지스틱 모델 구현**



먼저 로지스틱에 사용되는 시그모이드 함수를 만든다.

$$
\begin{align*}
\sigma(t) = \frac{1}{1 + e^{-t}}
\end{align*}
$$


```python
def logistic(logits):
    return 1.0 / (1 + np.exp(-logits))
```

가중치를 조정해나가기 위한 세타를 생성한다. 초기값은 랜덤이다.

여기에서 n은 특성이 두개이므로 2가된다.

$$
\begin{align*}
\hat y^{(i)} & = \theta^{T}\, \mathbf{x}^{(i)} \\
 & = \theta_0 + \theta_1\, \mathbf{x}_1^{(i)} + \cdots + \theta_n\, \mathbf{x}_n^{(i)}
\end{align*}
$$


```python
n_inputs = X_train.shape[1] #편향과 특성의 갯수
Theta = np.random.randn(n_inputs) #편향과 특성의 갯수만큼 세타값 랜덤초기화
```

**cost function 구현**

$$
\begin{align*}
(\boldsymbol{\theta}) = -\dfrac{1}{m} \sum\limits_{i=1}^{m}{\left[ y^{(i)} log\left(\hat{p}^{(i)}\right) + (1 - y^{(i)}) log\left(1 - \hat{p}^{(i)}\right)\right]}
\end{align*}
$$

위의 수식을 코드로 표현하면 다음과 같다.

```python
-np.mean(np.sum((y_train*np.log(Y_proba + epsilon) + (1-y_train)*np.log(1 - Y_proba + epsilon))))
```


배치 경사하강법 훈련은 아래 코드를 통해 이루어진다.

- `eta = 0.01`: 학습률
- `n_iterations = 5001` : 에포크 수
- `m = len(X_train)`: 훈련 세트 크기, 즉 훈련 샘플 수
- `epsilon = 1e-7`: $\log$ 값이 항상 계산되도록 더해지는 작은 실수
- `logits`: 모든 샘플에 대한 클래스별 점수, 즉 $\mathbf{X}_{\textit{train}}\, \Theta$
- `Y_proba`: 모든 샘플에 대해 계산된 클래스 별 소속 확률, 즉 $\hat P$


```python
#  배치 경사하강법 구현
eta = 0.1
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7

for iteration in range(n_iterations):     # 5001번 반복 훈련
    logits = X_train.dot(Theta)
    Y_proba = logistic(logits)
   
    if iteration % 500 == 0:
      loss = -np.mean(np.sum((y_train*np.log(Y_proba + epsilon) + (1-y_train)*np.log(1 - Y_proba + epsilon))))
      print(iteration, loss)
    
    error = Y_proba - y_train     # 그레이디언트 계산.
    gradients = 1/m * X_train.T.dot(error)
    
    Theta = Theta - eta * gradients
```

    0 79.35473984499612
    500 27.149524631560638
    1000 21.89438928577945
    1500 19.33777344771706
    2000 17.691444239326714
    2500 16.49516908325313
    3000 15.566000472955372
    3500 14.81327398979558
    4000 14.185530546071131
    4500 13.65075154805576
    5000 13.187653637231028
    


```python
Theta
```




    array([-10.56492618,   0.53611169,   4.82694082])



코드의 세부동작은 다음과 같다.

1.   logits = X_train.dot(Theta) 에서 행렬연산을 이용해 세타와 x값을 곱하여 logits 값을 얻는다.
2.   Y_proba = logistic(logits) 에 logits값을 시그모이드 함수에 넣어 Y_proba값을 얻는다.

3.   손실함수 계산을 통해 손실비용 loss를 얻는다.

4.   y의 확률값과 실제 y의값의 차이 error를 얻는다.

5.   이를 통해 gradient 계산을한다.

6.   세타에 학습률 * gradient만큼의 값을 빼서 세타값 재조정을 한뒤 다음 에포크로 넘어간다.



**검증**

위에서 얻은 세타값을 가지고
검증세트로 모델 성능을 판단한다.

Y_proba값이 0.5 이상이라면 버지니아로, 아니라면 버지니아가 아니라고 입력해준다.


```python
logits = X_valid.dot(Theta)              
Y_proba = logistic(logits)
y_predict = np.array([])
for i in range(len(Y_proba)):
  if Y_proba[i] >= 0.5:
    y_predict = np.append(y_predict, 1)
  else:
    y_predict = np.append(y_predict, 0)

accuracy_score = np.mean(y_predict == y_valid)  # 정확도 계산
accuracy_score
```




    0.9666666666666667



정확도는 위와같다.

이번엔 직접 데이터를 살펴보자.


```python
y_predict
```




    array([0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1.,
           0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.])




```python
y_valid
```




    array([0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0,
           1, 1, 0, 0, 0, 0, 0, 1])



**모델 규제**

일반적으로 l2규제를 사용한다.

코드에서 규제 작동 메커니즘은

$$
\begin{align*}
J(\boldsymbol{\theta}) & = \text{MSE}(\boldsymbol{\theta}) + \dfrac{\alpha}{2}\sum\limits_{i=1}^{n}{\theta_i}^2 \\
& = \text{MSE}(\boldsymbol{\theta}) + \dfrac{\alpha}{2}\left (\theta_1^2 + \cdots + \theta_n^2 \right )
\end{align*}
$$

수식의 계산된 값을 loss에 추가하고, 이를 gradient에 반영함으로써 이루어진다.


```python
eta = 0.1
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.5        # 규제 하이퍼파라미터

Theta = np.random.randn(n_inputs)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    logits = X_train.dot(Theta)
    Y_proba = logistic(logits)
    
    if iteration % 500 == 0:
        xentropy_loss = -np.mean(np.sum((y_train*np.log(Y_proba + epsilon) + (1-y_train)*np.log(1 - Y_proba + epsilon))))
        l2_loss = 1/2 * np.sum(np.square(Theta[1:]))  # 편향은 규제에서 제외
        loss = xentropy_loss + alpha * l2_loss        # l2 규제가 추가된 손실
        print(iteration, loss)
    
    error = Y_proba - y_train
    l2_loss_gradients = np.r_[np.zeros([1]), alpha * Theta[1:]]   # l2 규제 그레이디언트
    gradients = 1/m * X_train.T.dot(error) + l2_loss_gradients
    
    Theta = Theta - eta * gradients
```

    0 156.73838246234882
    500 36.11974638424874
    1000 34.306068180110614
    1500 34.02211206089248
    2000 33.9713877223945
    2500 33.96211929178583
    3000 33.96041878356459
    3500 33.960106551185575
    4000 33.96004921390298
    4500 33.96003868441418
    5000 33.96003675075696
    

다시 검증세트를 이용해 성능을 확인해보자.


```python
logits = X_valid.dot(Theta)              
Y_proba = logistic(logits)
y_predict = np.array([])
for i in range(len(Y_proba)):
  if Y_proba[i] >= 0.5:
    y_predict = np.append(y_predict, 1)
  else:
    y_predict = np.append(y_predict, 0)

accuracy_score = np.mean(y_predict == y_valid)  # 정확도 계산
accuracy_score
```




    0.8333333333333334



점수가 조금 떨어졌으나 중요한것은 테스트세트에 대한 성능이다.

이번에는 조기종료 기능을 추가한다.

조기종료는 검증세트에 대한 손실값이 이전 단계보다 커지면
바로 종료되는 기능이다. 이를 코드로 구현하면 다음과 같다.


```python
eta = 0.1 
n_iterations = 50000
m = len(X_train)
epsilon = 1e-7
alpha = 0.5            # 규제 하이퍼파라미터
best_loss = np.infty   # 최소 손실값 기억 변수

Theta = np.random.randn(n_inputs)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    # 훈련 및 손실 계산
    logits = X_train.dot(Theta)
    Y_proba = logistic(logits)
    error = Y_proba - y_train
    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1]), alpha * Theta[1:]]
    Theta = Theta - eta * gradients

    # 검증 세트에 대한 손실 계산
    logits = X_valid.dot(Theta)
    Y_proba = logistic(logits)
    xentropy_loss = -np.mean(np.sum((y_valid*np.log(Y_proba + epsilon) + (1-y_valid)*np.log(1 - Y_proba + epsilon))))
    l2_loss = 1/2 * np.sum(np.square(Theta[1:]))
    loss = xentropy_loss + alpha * l2_loss
    
    # 500 에포크마다 검증 세트에 대한 손실 출력
    if iteration % 500 == 0:
        print(iteration, loss)
        
    # 에포크마다 최소 손실값 업데이트
    if loss < best_loss:
        best_loss = loss
    else:                                      # 에포크가 줄어들지 않으면 바로 훈련 종료
        print(iteration - 1, best_loss)        # 종료되기 이전 에포크의 손실값 출력
        print(iteration, loss, "조기 종료!")
        break
```

    0 31.861885822665705
    500 12.54052343500867
    1000 11.955093912404864
    1500 11.865469014733975
    2000 11.849523658203214
    2500 11.846612260301496
    3000 11.846078169747395
    3500 11.845980107187028
    4000 11.845962099397736
    4500 11.845958792428052
    5000 11.84595818512936
    5500 11.845958073603674
    6000 11.84595805312284
    6500 11.845958049361693
    7000 11.845958048670985
    7500 11.845958048544144
    8000 11.845958048520849
    8351 11.845958048517204
    8352 11.845958048517206 조기 종료!
    

테스트 셋에 대하여 정확도를 판별한다.


```python
logits = X_test.dot(Theta)
Y_proba = logistic(logits)
y_predict = np.array([])
for i in range(len(Y_proba)):
  if Y_proba[i] >= 0.5:
    y_predict = np.append(y_predict, 1)
  else:
    y_predict = np.append(y_predict, 0)


accuracy_score = np.mean(y_predict == y_test)
accuracy_score
```




    0.9666666666666667



점수가 매우 높게나왔다.

이제 실제 로지스틱 모델과 얼마나 차이가 나는지 확인해보자.

**사이킷런 로지스틱 모델과 성능비교**


```python
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver="lbfgs", random_state=42)
log_reg.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=42, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)



기준이 정확도였으므로 성능 확인 기준역시 정확도로 둔다.

정확도를 구하는 법은 두가지가 있다.


```python
log_reg.score(X_test,y_test)
```




    0.9666666666666667




```python
from sklearn.metrics import accuracy_score
y_pred = log_reg.predict(X_test)
accuracy_score(y_test, y_pred)
```




    0.9666666666666667



실제 모델과 유사한 성능임을 확인할수 있다.

# 2.Logistic Regression 다중 클래스 분류 

앞에서 구현된 로지스틱 회귀 알고리즘에 일대다(OvR) 방식을 적용하여 붓꽃에 대한 다중 클래스 분류 알고리즘을 구현한다. 단, 사이킷런을 전혀 사용하지 않는다.



Rogistic Regression 모델로 다중 클래스 분류를 수행하기 위해서는 로지스틱 모델을 2개를 사용해야한다. 

먼저 setosa인지 아닌지를 판단하는 모델

그리고 virginica인지 아닌지를 판단하는 모델을 각각 만든후에

versicolor일 확률은 setosa와 virginica 둘다 아닌 확률로 계산해준다.


```python
X = iris["data"][:, (2, 3)]  # 꽃잎 길이, 꽃잎 넓이
y = iris["target"]
y0 = (iris["target"] == 0).astype(np.int) #setosa 판단 모델을 위한 데이터셋
y1 = (iris["target"] == 2).astype(np.int) #virginica 판단 모델을 위한 데이터셋
```


```python
X_with_bias = np.c_[np.ones([len(X), 1]), X] #편향추가
```


```python
np.random.seed(2042) #일정한 결과를 위해 랜덤시드 지정
```


```python
test_ratio = 0.2                                         # 테스트 세트 비율 = 20%
validation_ratio = 0.2                                   # 검증 세트 비율 = 20%
total_size = len(X_with_bias)                            # 전체 데이터셋 크기

test_size = int(total_size * test_ratio)                 # 테스트 세트 크기: 전체의 20%
validation_size = int(total_size * validation_ratio)     # 검증 세트 크기: 전체의 20%
train_size = total_size - test_size - validation_size    # 훈련 세트 크기: 전체의 60%
```


```python
rnd_indices = np.random.permutation(total_size) #데이터 섞기
```

모델 훈련은 각 클래스에 대해 각각 이루어지기 때문에
데이터셋도 개별적으로 준비해준다.


```python
X_train = X_with_bias[rnd_indices[:train_size]] 
y_train = y[rnd_indices[:train_size]]
y_train0 = y0[rnd_indices[:train_size]] #setosa에 대한 라벨
y_train1 = y1[rnd_indices[:train_size]] #virginica에 대한 라벨

X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]
y_valid0 = y0[rnd_indices[train_size:-test_size]] #setosa에 대한 검증세트 라벨
y_valid1 = y1[rnd_indices[train_size:-test_size]] #virginica에 대한 검증세트 라벨

X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]
```


```python
n_inputs = X_train.shape[1]
Theta0 = np.random.randn(n_inputs) #setosa 판단모델에 쓰이는 세타값
Theta1 = np.random.randn(n_inputs) #virginica 판단모델에 쓰이는 세타값
```

**setosa 판별 로지스틱 회귀 모델**


```python
eta = 0.1 
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.5            # 규제 하이퍼파라미터
best_loss0 = np.infty   # 최소 손실값 기억 변수

Theta0 = np.random.randn(n_inputs)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    # 훈련 및 손실 계산
    logits0 = X_train.dot(Theta0)
    Y_proba0 = logistic(logits0)
    error = Y_proba0 - y_train0
    gradients0 = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1]), alpha * Theta0[1:]]
    Theta0 = Theta0 - eta * gradients0

    # 검증 세트에 대한 손실 계산
    logits0 = X_valid.dot(Theta0)
    Y_proba0 = logistic(logits0)
    xentropy_loss0 = -np.mean(np.sum((y_valid0*np.log(Y_proba0 + epsilon) + (1-y_valid0)*np.log(1 - Y_proba0 + epsilon))))
    l2_loss0 = 1/2 * np.sum(np.square(Theta0[1:]))
    loss0 = xentropy_loss0 + alpha * l2_loss0
    
    # 500 에포크마다 검증 세트에 대한 손실 출력
    if iteration % 500 == 0:
        print(iteration, loss0)
        
    # 에포크마다 최소 손실값 업데이트
    if loss0 < best_loss0:
        best_loss0 = loss0
    else:                                      # 에포크가 줄어들지 않으면 바로 훈련 종료
        print(iteration - 1, best_loss0)        # 종료되기 이전 에포크의 손실값 출력
        print(iteration, loss0, "조기 종료!")
        break
```

    0 20.540019459712514
    500 7.744571615343959
    1000 7.672989036271927
    1500 7.668592640555666
    2000 7.668314272027711
    2500 7.668296612120626
    3000 7.668295491624586
    3500 7.668295420530142
    4000 7.668295416019264
    4500 7.668295415733049
    5000 7.668295415714894
    

**virginica 판별 로지스틱 회귀 모델**


```python
eta = 0.1 
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.5            # 규제 하이퍼파라미터
best_loss1 = np.infty   # 최소 손실값 기억 변수

Theta1 = np.random.randn(n_inputs)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    # 훈련 및 손실 계산
    logits1 = X_train.dot(Theta1)
    Y_proba1 = logistic(logits1)
    error = Y_proba1 - y_train1
    gradients1 = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1]), alpha * Theta1[1:]]
    Theta1 = Theta1 - eta * gradients1

    # 검증 세트에 대한 손실 계산
    logits1 = X_valid.dot(Theta1)
    Y_proba1 = logistic(logits1)
    xentropy_loss1 = -np.mean(np.sum((y_valid1*np.log(Y_proba1 + epsilon) + (1-y_valid1)*np.log(1 - Y_proba1 + epsilon))))
    l2_loss1 = 1/2 * np.sum(np.square(Theta1[1:]))
    loss1 = xentropy_loss1 + alpha * l2_loss1
    
    # 500 에포크마다 검증 세트에 대한 손실 출력
    if iteration % 500 == 0:
        print(iteration, loss1)
        
    # 에포크마다 최소 손실값 업데이트
    if loss1 < best_loss1:
        best_loss1 = loss1
    else:                                      # 에포크가 줄어들지 않으면 바로 훈련 종료
        print(iteration - 1, best_loss1)        # 종료되기 이전 에포크의 손실값 출력
        print(iteration, loss1, "조기 종료!")
        break
```

    0 45.38818486389959
    500 12.482904005693054
    1000 11.947222069327108
    1500 11.864096195806566
    2000 11.849273910674974
    2500 11.846566475123907
    3000 11.846069764314986
    3500 11.845978563684064
    4000 11.845961815948371
    4500 11.845958740374874
    5000 11.845958175570198
    

**이제 테스트셋에 적용해본다.**

위에서 구한 두개의 세타값을 이용해

1.   setosa일 확률(setosa_proba)
2.   virginica일 확률(virginica_proba)
3.   versicolor일 확률(1 - setosa_proba - virginica_proba)

셋중에 가장 높은것을 채택하여 분류를 진행한다.



```python
logits = X_test.dot(Theta0) #setosa에 대한 확률값 추정  
setosa_proba = logistic(logits)

logits = X_test.dot(Theta1) #virginica에 대한 확률값 추정 
virginica_proba = logistic(logits)

y_predict = np.array([])
for i in range(len(Y_proba0)):
  prob_list = [[setosa_proba[i], 0], [1-setosa_proba[i]-virginica_proba[i], 1], [virginica_proba[i], 2]]
  prob_list.sort(reverse=True) #가장 높은 확률이 가장 앞으로 오게끔 정렬해준다.
  y_predict = np.append(y_predict, prob_list[0][1]) #가장 확률이 높았던 것을 예측값으로 결정한다.
```


```python
accuracy_score = np.mean(y_predict == y_test)
accuracy_score
```




    0.9333333333333333



약 0.93점으로 괜찮은 점수가 나왔다.

이제 사이킷런의 로지스틱 모델과의 성능을 비교해보자.

모델의 solver 값을 'newton-cg'로 주면 multinomial logistic regression모델을 세울수있다.


```python
from sklearn.linear_model import LogisticRegression
multi_log_reg = LogisticRegression(solver='newton-cg', random_state=42).fit(X_train,y_train)

multi_log_reg.score(X_test,y_test)
```




    0.9333333333333333



직접 구현한 코드와 사이킷런에 내장되어있는 로지스틱 모델과 성능이 같음을 확인할수 있었다.

과제2를 하며 왜인지는 모르겠지만

직접 구현한 모델이나 사이킷런의 로지스틱모델이 versicolor에 대해 분류를 잘못했다.

규제값도 바꿔가며 고민해봤지만 이유를 찾지는 못하였으므로

setosa와 virginica에대해 모델을 설정하였다. 결과는 매우 좋게나왔다.

# 3.사진 분류

A. 사진을 낮과 밤으로 분류하는 로지스틱 회귀 모델을 구현하라.

B. 자신의 알고리즘과 사이킷런에서 제공하는 LogisticRegression 모델의 성능을 비교하라.

C. 사진을 낮과 밤, 실내와 실외로 분류하는 다중 레이블 분류 모델을 두 개의 로지스틱 회귀 모델을 이용하여 구현하라.

단, 모델 구현에 필요한 사진을 직접 구해야 한다. 최소 100장 이상의 사진 활용해야 한다.

A에서 직접 구현한 모델과 사이킷런의 로지스틱 모델의 성능비교를 하기위해 순서를 바꿔 A->C->B 순으로 진행한다.

## A

사진을 낮과 밤으로 분류하는 로지스틱 회귀 모델 구현

직접 수집한 이미지를 구글드라이브를 통해 다운로드받기


```python
from urllib import request
url = "https://docs.google.com/uc?export=download&id=1emB4lSxEzxzEt7_20w2DZ_Dw1sYS1grA"
request.urlretrieve(url,"day_night.zip")
```




    ('day_night.zip', <http.client.HTTPMessage at 0x7f9216457510>)



파일을 압축해제하기


```python
import os
import zipfile

local_zip = '/content/day_night.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/content')
zip_ref.close()
```

작업에 필요한 모듈 임포트


```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2 
import os 
from random import shuffle 
from tqdm import tqdm 
from PIL import Image
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.filterwarnings('ignore')
import os
# Any results you write to the current directory are saved as output.
```

label과 train, test에 따라 경로 지정하기


```python
train_day = "day_night/train/day"
train_night= "day_night/train/night"
test_day= "day_night/test/day"
test_night= "day_night/test/night"
image_size = 128
```

위의 과정이 제대로 되었는지 확인하기 위해 시험삼아 이미지를 불러온다.


```python
Image.open("day_night/train/day/day_120.jpg")
```




    
![output_80_0](https://user-images.githubusercontent.com/42956142/117577881-a7fa3b80-b126-11eb-86b4-e8f4059ae919.png)
    




```python
Image.open("day_night/train/night/night_120.jpg")
```




    
![output_81_0](https://user-images.githubusercontent.com/42956142/117577931-eb54aa00-b126-11eb-8327-0994b58f3c44.png)
    



수집되어있는 사진들은 사이즈가 모두 제각각이다.

머신러닝은 사진의 크기에 따라 특성수를 다르게 받아들이기때문에
이를 조정해주는 작업이 필요하다.

이를 resize라하고, 코드는 아래와 같다.


```python
for image in tqdm(os.listdir(train_night)): 
    path = os.path.join(train_night, image)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    img = cv2.resize(img, (image_size, image_size)).flatten()   
    np_img=np.asarray(img)
    
for image2 in tqdm(os.listdir(train_day)): 
    path = os.path.join(train_day, image2)
    img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    img2 = cv2.resize(img2, (image_size, image_size)).flatten() 
    np_img2=np.asarray(img2)

plt.figure(figsize=(10,10))
plt.subplot(1, 2, 1)
plt.imshow(np_img.reshape(image_size, image_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(np_img2.reshape(image_size, image_size))
plt.axis('off')
plt.title("day and night in GrayScale")
```

    100%|██████████| 400/400 [00:00<00:00, 1378.18it/s]
    100%|██████████| 400/400 [00:00<00:00, 1439.68it/s]
    




    Text(0.5, 1.0, 'day and night in GrayScale')




    
![output_83_2](https://user-images.githubusercontent.com/42956142/117577939-f7d90280-b126-11eb-893f-5c5444678b77.png)
    


경로에 따라 나뉘어져 있는 낮과 밤사진들을

하나의 트레이닝 셋으로 합쳐주는 과정이 필요하다.

이 과정에 데이터 라벨링이 완료된다.


```python
def train_data():
    train_data_night = [] 
    train_data_day=[]
    for image1 in tqdm(os.listdir(train_night)): 
        path = os.path.join(train_night, image1)
        img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img1 = cv2.resize(img1, (image_size, image_size))
        train_data_night.append(img1) 
    for image2 in tqdm(os.listdir(train_day)): 
        path = os.path.join(train_day, image2)
        img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img2 = cv2.resize(img2, (image_size, image_size))
        train_data_day.append(img2) 
    
    train_data= np.concatenate((np.asarray(train_data_night),np.asarray(train_data_day)),axis=0)
    return train_data 
```

같은 작업을 테스트셋에 대해서도 해준다.


```python
def test_data():
    test_data_night = [] 
    test_data_day=[]
    for image1 in tqdm(os.listdir(test_night)): 
        path = os.path.join(test_night, image1)
        img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img1 = cv2.resize(img1, (image_size, image_size))
        test_data_night.append(img1) 
    for image2 in tqdm(os.listdir(test_day)): 
        path = os.path.join(test_day, image2)
        img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img2 = cv2.resize(img2, (image_size, image_size))
        test_data_day.append(img2) 
    
    test_data= np.concatenate((np.asarray(test_data_night),np.asarray(test_data_day)),axis=0) 
    return test_data 
```

이제 트레인셋과 테스트셋 설정 해준다.

아래의 과정에서 features와 label을 분리하여 저장한다.


```python
train_data = train_data() 
test_data = test_data()
```

    100%|██████████| 400/400 [00:00<00:00, 1176.50it/s]
    100%|██████████| 400/400 [00:00<00:00, 1580.24it/s]
    100%|██████████| 100/100 [00:00<00:00, 1831.12it/s]
    100%|██████████| 100/100 [00:00<00:00, 1568.23it/s]
    


```python
x_data=np.concatenate((train_data,test_data),axis=0)
x_data = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
```


```python
z1 = np.zeros(400)
o1 = np.ones(400)
Y_train = np.concatenate((o1, z1), axis=0)
z = np.zeros(100)
o = np.ones(100)
Y_test = np.concatenate((o, z), axis=0)
```


```python
y_data=np.concatenate((Y_train,Y_test),axis=0).reshape(x_data.shape[0],1)
```


```python
print("X shape: " , x_data.shape)
print("Y shape: " , y_data.shape)
```

    X shape:  (1000, 128, 128)
    Y shape:  (1000, 1)
    

사이킷런의 train_test_spilit 활용
train, test셋 분리하기.


```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=42)
number_of_train = x_train.shape[0]
number_of_test = x_test.shape[0]
```


```python
x_train_flatten = x_train.reshape(number_of_train,x_train.shape[1]*x_train.shape[2])
x_test_flatten = x_test .reshape(number_of_test,x_test.shape[1]*x_test.shape[2])
print("X train flatten",x_train_flatten.shape)
print("X test flatten",x_test_flatten.shape)
```

    X train flatten (850, 16384)
    X test flatten (150, 16384)
    


```python
x_train = x_train_flatten.T
x_test = x_test_flatten.T
y_test = y_test.T
y_train = y_train.T
day_night_y_test = y_test
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)
```

    x train:  (16384, 850)
    x test:  (16384, 150)
    y train:  (1, 850)
    y test:  (1, 150)
    

데이터 전처리가 완료되었다.

다음으로 로지스틱 모델을 직접 구현해준다.


```python
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w, b

def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    return cost,gradients

def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    for i in range(number_of_iterarion):
        
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 100 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list

def predict(w,b,x_test):
    
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))

    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    dimension =  x_train.shape[0]
    w,b = initialize_weights_and_bias(dimension)

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)
    
    print("Test Accuracy: {} %".format(round(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100,2)))
    print("Train Accuracy: {} %".format(round(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100,2)))
```

에포크는 1500으로, 학습률을 0.01으로 지정한뒤에 학습을 시작한다.


```python
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 1500)
```

    Cost after iteration 0: nan
    Cost after iteration 100: 0.493764
    Cost after iteration 200: 0.452712
    Cost after iteration 300: 0.457164
    Cost after iteration 400: 0.414077
    Cost after iteration 500: 0.411219
    Cost after iteration 600: 0.409035
    Cost after iteration 700: 0.376400
    Cost after iteration 800: 0.346463
    Cost after iteration 900: 0.319091
    Cost after iteration 1000: 0.293898
    Cost after iteration 1100: 0.270654
    Cost after iteration 1200: 0.249235
    Cost after iteration 1300: 0.229576
    Cost after iteration 1400: 0.211625
    


    
![output_101_1](https://user-images.githubusercontent.com/42956142/117577949-02939780-b127-11eb-9ec2-0abf6f1c51c9.png)
    


    Test Accuracy: 78.67 %
    Train Accuracy: 95.76 %
    

train_set 비해 test_set에 대한 성능이 낮게 나왔음이 확인되었고,
과대적합이 의심된다.

## B
구현한 자신의 알고리즘과 사이킷런에서 제공하는 LogisticRegression 모델의 성능을 비교하라.

사이킷런 LogisticRegression에 넣기위해 데이터의 형태를 맞춰준다.


```python
x_train.shape
```




    (16384, 850)




```python
y_train.shape
```




    (1, 850)




```python
y_train2 = np.array([])
for i in y_train:
  y_train2 = np.append(y_train2, np.array([i]))
```


```python
y_test2 = np.array([])
for i in y_test:
  y_test2 = np.append(y_test2, np.array([i]))
```


```python
y_train2.shape
```




    (850,)



slover값을 'saga'로 지정, multi_class를 'multinomial'로 지정하면 로지스틱모델을 이용할수있다.


```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(penalty='none', 
                         tol=0.1, solver='saga',
                         multi_class='multinomial').fit(x_train.T, y_train2)
```


```python
clf.score(x_test.T, y_test2)
```




    0.7666666666666667




```python
pred1 = clf.predict(x_test.T)
```

직접 구현한 로지스틱 모델의 정확도는 약 78%
사이킷런에 내장된 로지스틱 모델의 정확도는 약 76%로

둘의 성능은 유사하나 사이킷런의 LogisiticRegression 모델이 좀더 우수함을 알수있다.

## C

사진을 낮과 밤, 실내와 실외로 분류하는 다중 레이블 분류 모델을 두 개의 로지스틱 회귀 모델을 이용하여 구현하라.

두 개의 로지스틱 회귀모델중, 낮과 밤을 분류하는 모델은 위에서 이미 만들었으므로

여기에서는 먼저 실내와 실외를 분류하는 로지스틱 모델을 만들도록 한다.

과정은 낮과 밤을 분류할때와 거의 유사하므로 
비슷한부분은 아주 간단하게만 집고 넘어간다.

구글 드라이브에서 실내와 실외의 데이터 다운로드


```python
from urllib import request
url = "https://docs.google.com/uc?export=download&id=1CPbsXHOxFEAic3YQBxDdTDKEZXkXMdP1"
request.urlretrieve(url,"indoor_outdoor.zip")
```




    ('indoor_outdoor.zip', <http.client.HTTPMessage at 0x7f92162ae290>)



압축풀기


```python
import os
import zipfile

local_zip = '/content/indoor_outdoor.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/content')
zip_ref.close()
```

경로지정


```python
train_indoor = "indoor_outdoor/train/indoors"
train_outdoor= "indoor_outdoor/train/outdoors"
test_indoor= "indoor_outdoor/test/indoors"
test_outdoor= "indoor_outdoor/test/outdoors"
image_size = 128
```

제대로 다운로드가 완료되었는지 체크


```python
Image.open("indoor_outdoor/train/indoors/indoors.101.jpg")
```




    
![output_124_0](https://user-images.githubusercontent.com/42956142/117577958-0e7f5980-b127-11eb-9b86-030e7069c2fe.png)
    




```python
Image.open("indoor_outdoor/train/outdoors/outdoors_120.jpg")
```




    
![output_125_0](https://user-images.githubusercontent.com/42956142/117577964-17702b00-b127-11eb-9cf0-c4a05c531663.png)




사진 리사이즈


```python
for image in tqdm(os.listdir(train_indoor)): 
    path = os.path.join(train_indoor, image)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    img = cv2.resize(img, (image_size, image_size)).flatten()   
    np_img=np.asarray(img)
    
for image2 in tqdm(os.listdir(train_outdoor)): 
    path = os.path.join(train_outdoor, image2)
    img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    img2 = cv2.resize(img2, (image_size, image_size)).flatten() 
    np_img2=np.asarray(img2)

plt.figure(figsize=(10,10))
plt.subplot(1, 2, 1)
plt.imshow(np_img.reshape(image_size, image_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(np_img2.reshape(image_size, image_size))
plt.axis('off')
plt.title("indoor and outdoor in GrayScale")
```

    100%|██████████| 400/400 [00:00<00:00, 1659.49it/s]
    100%|██████████| 400/400 [00:00<00:00, 1553.95it/s]
    




    Text(0.5, 1.0, 'indoor and outdoor in GrayScale')




    
![output_127_2](https://user-images.githubusercontent.com/42956142/117577973-20f99300-b127-11eb-8475-935b0400f386.png)



트레인/데이터셋 구성하기


```python
def train_data():
    train_data_indoor = [] 
    train_data_outdoor=[]
    for image1 in tqdm(os.listdir(train_indoor)): 
        path = os.path.join(train_indoor, image1)
        img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img1 = cv2.resize(img1, (image_size, image_size))
        train_data_indoor.append(img1) 
    for image2 in tqdm(os.listdir(train_outdoor)): 
        path = os.path.join(train_outdoor, image2)
        img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img2 = cv2.resize(img2, (image_size, image_size))
        train_data_outdoor.append(img2) 
    
    train_data= np.concatenate((np.asarray(train_data_indoor),np.asarray(train_data_outdoor)),axis=0)
    return train_data 
```


```python
def test_data():
    test_data_indoor = [] 
    test_data_outdoor=[]
    for image1 in tqdm(os.listdir(test_indoor)): 
        path = os.path.join(test_indoor, image1)
        img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img1 = cv2.resize(img1, (image_size, image_size))
        test_data_indoor.append(img1) 
    for image2 in tqdm(os.listdir(test_outdoor)): 
        path = os.path.join(test_outdoor, image2)
        img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img2 = cv2.resize(img2, (image_size, image_size))
        test_data_outdoor.append(img2) 
    
    test_data= np.concatenate((np.asarray(test_data_indoor),np.asarray(test_data_outdoor)),axis=0) 
    return test_data 
```


```python
train_data = train_data() 
test_data = test_data()
```

    100%|██████████| 400/400 [00:00<00:00, 1323.72it/s]
    100%|██████████| 400/400 [00:00<00:00, 1624.08it/s]
    100%|██████████| 100/100 [00:00<00:00, 1562.70it/s]
    100%|██████████| 100/100 [00:00<00:00, 1554.97it/s]
    


```python
x_data=np.concatenate((train_data,test_data),axis=0)
x_data = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
```


```python
z1 = np.zeros(400)
o1 = np.ones(400)
Y_train = np.concatenate((o1, z1), axis=0)
z = np.zeros(100)
o = np.ones(100)
Y_test = np.concatenate((o, z), axis=0)
```


```python
y_data=np.concatenate((Y_train,Y_test),axis=0).reshape(x_data.shape[0],1)
```


```python
print("X shape: " , x_data.shape)
print("Y shape: " , y_data.shape)
```

    X shape:  (1000, 128, 128)
    Y shape:  (1000, 1)
    

사이킷런 train_test_split을 이용해
트레이닝셋과 테스트셋 분리하기


```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=42)
number_of_train = x_train.shape[0]
number_of_test = x_test.shape[0]
```


```python
x_train_flatten = x_train.reshape(number_of_train,x_train.shape[1]*x_train.shape[2])
x_test_flatten = x_test .reshape(number_of_test,x_test.shape[1]*x_test.shape[2])
print("X train flatten",x_train_flatten.shape)
print("X test flatten",x_test_flatten.shape)
```

    X train flatten (850, 16384)
    X test flatten (150, 16384)
    


```python
x_train = x_train_flatten.T
x_test = x_test_flatten.T
y_test = y_test.T
y_train = y_train.T
out_doors_y_test = y_test
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)
```

    x train:  (16384, 850)
    x test:  (16384, 150)
    y train:  (1, 850)
    y test:  (1, 150)
    

로지스틱 모델 구현


```python
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w, b

def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    return cost,gradients

def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    for i in range(number_of_iterarion):
        
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 100 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list

def predict(w,b,x_test):
    
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))

    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    dimension =  x_train.shape[0]
    w,b = initialize_weights_and_bias(dimension)

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)
    
    print("Test Accuracy: {} %".format(round(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100,2)))
    print("Train Accuracy: {} %".format(round(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100,2)))
```

학습시작


```python
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 1500)
```

    Cost after iteration 0: nan
    Cost after iteration 100: 3.499697
    Cost after iteration 200: 3.608301
    Cost after iteration 300: 3.318072
    Cost after iteration 400: 3.163057
    Cost after iteration 500: 3.053198
    Cost after iteration 600: 2.962918
    Cost after iteration 700: 2.880539
    Cost after iteration 800: 2.802327
    Cost after iteration 900: 2.728196
    Cost after iteration 1000: 2.657614
    Cost after iteration 1100: 2.589571
    Cost after iteration 1200: 2.523354
    Cost after iteration 1300: 2.458452
    Cost after iteration 1400: 2.394409
    


    
![output_143_1](https://user-images.githubusercontent.com/42956142/117577990-2a82fb00-b127-11eb-9128-552dd1da0b03.png)



    Test Accuracy: 58.0 %
    Train Accuracy: 63.06 %
    

결과가 매우 많이 좋지않다.

58%면 분류 모델의 존재이유가 의심된다.

학습에 충분한 데이터를 모으지 못했던게 원인으로 생각된다.

이번엔 마찬가지로 사이킷런과 성능비교를 해본다.


```python
in_out_y_train = np.array([])
for i in y_train:
  in_out_y_train = np.append(in_out_y_train, np.array([i]))
```


```python
in_out_y_test = np.array([])
for i in y_test:
  in_out_y_test = np.append(in_out_y_test, np.array([i]))
```


```python
in_out_y_test
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 1., 1.,
           1., 0., 1., 0., 1., 1., 0., 1., 1., 1., 0., 0., 0., 1., 1., 0., 0.,
           1., 1., 0., 0., 0., 1., 1., 1., 1., 0., 1., 1., 0., 0., 0., 1., 0.,
           1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 1., 0., 1., 0., 0., 0., 1.,
           1., 0., 1., 0., 0., 0., 1., 1., 0., 1., 0., 1., 0., 1., 1., 1., 1.,
           0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 1.,
           0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 1., 0., 1., 0., 0.,
           1., 1., 0., 0., 1., 1., 0., 0., 0., 1., 1., 0., 1., 1., 0., 0., 1.,
           1., 1., 1., 1., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1.])




```python
from sklearn.linear_model import LogisticRegression

lg2 = LogisticRegression(penalty='none', 
                         tol=0.1, solver='saga',C = 0.5,
                         multi_class='multinomial').fit(x_train.T, in_out_y_train)
```


```python
pred2 = lg2.predict(x_test.T)
```


```python
lg2.score(x_test.T, in_out_y_test)
```




    0.6733333333333333



성능이 직접구현한것보다 좋게나왔지만 여전히 매우 좋지 않는 수치이다.

이로써 두개의 모델이 모두 준비되었으므로, 두개의 예측값을 합쳐서 하나의 array로 만들어 다중 라벨 분류를 완성한다.

마찬가지로 낮과밤, 실내실외의 라벨이 합쳐진 테스트셋과 비교하여 정확도 성능을 측정한다.


```python
multi_label_list = []
for i in range(len(pred1)):
 multi_label_list.append([pred1[i], pred2[i]]) # 낮과밤에 대한 예측결과와 실내실외에 대한 예측결과를 샘플별로 묶어서 리스트에 저장한다.
```


```python
multi_label_pred = np.array(multi_label_list) # 저장된 리스트를 array로 바꾼다.
```


```python
multi_label_test_list = []
for i in range(len(out_doors_y_test)):
 multi_label_test_list.append([day_night_y_test[0][i], out_doors_y_test[0][i]]) # 낮과밤, 실내실외에 대한 정답을 샘플별로 묶어서 리스트에 저장한다.
```


```python
multi_label_y_test = np.array(multi_label_test_list) # 저장된 리스트를 array로 바꿔준다.
```

이제 마지막으로 정확도를 측정한다.


```python
accuracy_score = np.mean(multi_label_pred == multi_label_y_test)
accuracy_score
```




    0.5333333333333333



약 53%로 다중분류모델의 성능이 매우 좋지 않았음을 확인하였다.

규제값을 이리저리 바꿔보기도 했지만 큰 성능향상이 있지 않았고,

에포크수도 많이늘려보았지만 과대적합만 발생할뿐 이렇다할 변화가 있지는 않았다.

학습에 충분히 필요한 데이터를 모으지 못했던것, 그리고 좋음 품질의 샘플을 모으지 못했던것이

낮은 성능의 가장 큰 이유라고 생각된다.
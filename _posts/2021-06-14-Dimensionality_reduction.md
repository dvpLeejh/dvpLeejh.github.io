---
title: "차원축소 알고리즘 성능비교하기"
tags:
  - Machine Learning
  - Project

categories:
    - Machine Learning

last_modified_at: 2021-06-14

use_math: true

toc: true
toc_sticky: true

---

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

**project 8**

차원 축소의 각 알고리즘을 활용하여 데이터를 전처리한 후
분류모델을 통해 각 알고리즘의 성능 확인하기

차원축소는 2차원과 3차원으로 나누어 진행한다.

대상 알고리즘

*   tsne
*   pca
*   LLE
*   mds

먼저 알고리즘을 시각적으로 표시하기 위해 plot_digits 도구를 만든다.


```python
from sklearn.preprocessing import MinMaxScaler
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

def plot_digits(X, y, min_distance=0.05, images=None, figsize=(13, 10)):
    # Let's scale the input features so that they range from 0 to 1
    X_normalized = MinMaxScaler().fit_transform(X)
    # Now we create the list of coordinates of the digits plotted so far.
    # We pretend that one is already plotted far away at the start, to
    # avoid `if` statements in the loop below
    neighbors = np.array([[10., 10.]])
    # The rest should be self-explanatory
    plt.figure(figsize=figsize)
    cmap = mpl.cm.get_cmap("jet")
    digits = np.unique(y)
    for digit in digits:
        plt.scatter(X_normalized[y == digit, 0], X_normalized[y == digit, 1], c=[cmap(digit / 9)])
    plt.axis("off")
    ax = plt.gcf().gca()  # get current axes in current figure
    for index, image_coord in enumerate(X_normalized):
        closest_distance = np.linalg.norm(np.array(neighbors) - image_coord, axis=1).min()
        if closest_distance > min_distance:
            neighbors = np.r_[neighbors, [image_coord]]
            if images is None:
                plt.text(image_coord[0], image_coord[1], str(int(y[index])),
                         color=cmap(y[index] / 9), fontdict={"weight": "bold", "size": 16})
            else:
                image = images[index].reshape(28, 28)
                imagebox = AnnotationBbox(OffsetImage(image, cmap="binary"), image_coord)
                ax.add_artist(imagebox)
```

이번에 활용할 데이터는 MNIST 데이터이다.


```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
```

MNIST에는 약 7만개의 데이터가 있고 이를 전부다 활용하면

매우 많은 시간이 소요되므로 여기에서는 만개의 데이터만을 활용하여

성능을 측정할것이다.


```python
mnist.target = mnist.target.astype(np.uint8)

np.random.seed(42)

m = 10000
idx = np.random.permutation(60000)[:m]

X = mnist['data'][idx]
y = mnist['target'][idx]
```

# 2차원

**TSNE**


```python
from sklearn.manifold import TSNE
import time

t0 = time.time()
tsne = TSNE(n_components=2, random_state=42)
X_tsne_reduced = tsne.fit_transform(X)
t1 = time.time()
print("t-SNE took {:.1f}s.".format(t1 - t0))
plot_digits(X_tsne_reduced, y)
plt.show()
```

    t-SNE took 274.7s.
    


    
![output_11_1](https://user-images.githubusercontent.com/42956142/121888359-dd5ffd80-cd52-11eb-8253-da81415b7af0.png)

    


**PCA**


```python
from sklearn.decomposition import PCA
import time

t0 = time.time()
pca = PCA(n_components=2, random_state=42)
X_pca_reduced = pca.fit_transform(X)
t1 = time.time()
print("PCA took {:.1f}s.".format(t1 - t0))
plot_digits(X_pca_reduced, y)
plt.show()
```

    PCA took 1.1s.
    


    
![output_13_1](https://user-images.githubusercontent.com/42956142/121888434-f2d52780-cd52-11eb-8f6a-bf44bcaff2ae.png)
    


**지역적 선형 임베딩(LLE)**


```python
from sklearn.manifold import LocallyLinearEmbedding

t0 = time.time()
lle = LocallyLinearEmbedding(n_components=2, random_state=42)
X_lle_reduced = lle.fit_transform(X)
t1 = time.time()
print("LLE took {:.1f}s.".format(t1 - t0))
plot_digits(X_lle_reduced, y)
plt.show()
```

    LLE took 199.5s.
    


    
![output_15_1](https://user-images.githubusercontent.com/42956142/121888458-fbc5f900-cd52-11eb-8503-ffa6e538d42d.png)
    


**MDS**


```python
from sklearn.manifold import MDS

m = 2000
t0 = time.time()
mds = MDS(n_components=2, random_state=42)
X_mds_reduced = mds.fit_transform(X[:m])
t1 = time.time()
print("MDS took {:.1f}s (on just 2,000 MNIST images instead of 10,000).".format(t1 - t0))
plot_digits(X_mds_reduced, y[:m])
plt.show()
```

    MDS took 155.6s (on just 2,000 MNIST images instead of 10,000).
    


    
![output_17_1](https://user-images.githubusercontent.com/42956142/121888478-02547080-cd53-11eb-9464-161153e5af73.png)
    


위의 그림들은 각 알고리즘들의 전처리 결과를 시각적으로 표현한 것이다.

이제 전치리가 완료된 데이터를 가지고 RandomForestClassifier을 활용하여

분류를 진행한뒤, 정확도에 따라 알고리즘에 따라 전처리가 얼마나 잘 되었는지

측정할 것이다.


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```

훈련에 사용할 데이터는 8천개, 테스트에 사용할 데이터는 2천개를 사용할 것이다.

(단, mds는 시간상의 관계로 2천개의 데이터만 전처리를 진행하여 훈련에는 1600개를, 학습에는 400개의 데이터를 사용한다)


```python
X_train_tsne = X_tsne_reduced[:8000]
X_train_pca = X_pca_reduced[:8000]
X_train_lle = X_lle_reduced[:8000]
X_train_mds = X_mds_reduced[:1600]
y_train = y[:8000]
y_train_mds = y[:1600]

X_test_tsne = X_tsne_reduced[8000:]
X_test_pca = X_pca_reduced[8000:]
X_test_lle = X_lle_reduced[8000:]
X_test_mds = X_mds_reduced[1600:]
y_test = y[8000:]
y_test_mds = y[1600:2000]
```

TSNE로 차원축소된 데이터 학습 & 평가


```python
rnd_clf_tsne = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf_tsne.fit(X_train_tsne, y_train)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=42, verbose=0,
                           warm_start=False)




```python
y_pred_tsne = rnd_clf_tsne.predict(X_test_tsne)
accuracy_score(y_test, y_pred_tsne)
```




    0.948



PCA 기법으로 차원축소된 데이터 학습 & 평가


```python
rnd_clf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf_pca.fit(X_train_pca, y_train)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=42, verbose=0,
                           warm_start=False)




```python
y_pred_pca = rnd_clf_pca.predict(X_test_pca)
accuracy_score(y_test, y_pred_pca)
```




    0.39



지역적 선형 임베딩(LLE)로 차원축소된 데이터 학습 & 평가


```python
rnd_clf_lle = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf_lle.fit(X_train_lle, y_train)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=42, verbose=0,
                           warm_start=False)




```python
y_pred_lle = rnd_clf_lle.predict(X_test_lle)
accuracy_score(y_test, y_pred_lle)
```




    0.8515



MDS로 차원축소된 데이터 학습 & 평가


```python
rnd_clf_mds = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf_mds.fit(X_train_mds, y_train_mds)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=42, verbose=0,
                           warm_start=False)




```python
y_pred_mds = rnd_clf_mds.predict(X_test_mds)
accuracy_score(y_test_mds, y_pred_mds)
```




    0.4775



결과는 

*   tsne기법 사용 : 94.8%
*   pca기법 사용 : 39%
*   lle기법 사용 : 85.15%
*   mds기법 사용 : 47.75%

tsne를 사용하여 전처리를 진행했을때

RandomForestClassifier가 가장 분류를 잘하였다.

# 3차원

이번에는 2차원이 아닌 3차원으로 차원축소를 진행한다.

TSNE


```python
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

t0 = time.time()
tsne = TSNE(n_components=3, random_state=42)
X_tsne_reduced = tsne.fit_transform(X)
t1 = time.time()
print("t-SNE took {:.1f}s.".format(t1 - t0))

X_normalized_tsne = MinMaxScaler().fit_transform(X_tsne_reduced)
fig = plt.figure(figsize=(13,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_normalized_tsne[:, 0], X_normalized_tsne[:, 1], X_normalized_tsne[:, 2], c=y, cmap="jet")
plt.show()
```

    t-SNE took 623.7s.
    


    
![output_38_1](https://user-images.githubusercontent.com/42956142/121888515-10a28c80-cd53-11eb-86d3-7c49d65531bb.png)
    


PCA


```python
from sklearn.decomposition import PCA
import time

t0 = time.time()
pca = PCA(n_components=3, random_state=42)
X_pca_reduced = pca.fit_transform(X)
t1 = time.time()
print("PCA took {:.1f}s.".format(t1 - t0))

X_normalized_pca = MinMaxScaler().fit_transform(X_pca_reduced)
fig = plt.figure(figsize=(13,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_normalized_pca[:, 0], X_normalized_pca[:, 1], X_normalized_pca[:, 2], c=y, cmap="jet")
plt.show()
```

    PCA took 1.3s.
    


    
![output_40_1](https://user-images.githubusercontent.com/42956142/121888534-17310400-cd53-11eb-8497-9b6fcd98b833.png)
    


지역적 선형 임베딩(LLE)


```python
from sklearn.manifold import LocallyLinearEmbedding

t0 = time.time()
lle = LocallyLinearEmbedding(n_components=3, random_state=42)
X_lle_reduced = lle.fit_transform(X)
t1 = time.time()
print("LLE took {:.1f}s.".format(t1 - t0))

X_normalized_lle = MinMaxScaler().fit_transform(X_lle_reduced)
fig = plt.figure(figsize=(13,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_normalized_lle[:, 0], X_normalized_lle[:, 1], X_normalized_lle[:, 2], c=y, cmap="jet")
plt.show()
```

    LLE took 197.1s.
    


    
![output_42_1](https://user-images.githubusercontent.com/42956142/121888564-1e581200-cd53-11eb-9c22-164e4e4a6783.png)
    


MDS


```python
from sklearn.manifold import MDS

m = 2000
t0 = time.time()
mds = MDS(n_components=3, random_state=42)
X_mds_reduced = mds.fit_transform(X[:m])
t1 = time.time()
print("MDS took {:.1f}s (on just 2,000 MNIST images instead of 10,000).".format(t1 - t0))

X_normalized_mds = MinMaxScaler().fit_transform(X_mds_reduced)
fig = plt.figure(figsize=(13,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_normalized_mds[:, 0], X_normalized_mds[:, 1], X_normalized_mds[:, 2], c=y[:2000], cmap="jet")
plt.show()
```

    MDS took 144.0s (on just 2,000 MNIST images instead of 10,000).
    


    
![output_44_1](https://user-images.githubusercontent.com/42956142/121888587-24e68980-cd53-11eb-9b9a-7378a66687f8.png)
    


마찬가지로 시간과 비용상의 문제로 10000개의 데이터만 가지고 전처리를 진행하였다.

아까와 같이 훈련용/테스트용 데이터로 분리해준다.


```python
X_train_tsne = X_tsne_reduced[:8000]
X_train_pca = X_pca_reduced[:8000]
X_train_lle = X_lle_reduced[:8000]
X_train_mds = X_mds_reduced[:1600]
y_train = y[:8000]
y_train_mds = y[:1600]

X_test_tsne = X_tsne_reduced[8000:]
X_test_pca = X_pca_reduced[8000:]
X_test_lle = X_lle_reduced[8000:]
X_test_mds = X_mds_reduced[1600:]
y_test = y[8000:]
y_test_mds = y[1600:2000]
```

이제 다시 각 알고리즘에 따라 차원축소된 데이터를 가지고 
RandomForestClassifier를 이용하여 학습&평가를 진행한다.

TSNE으로 차원축소된 데이터 학습 & 평가


```python
rnd_clf_tsne = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf_tsne.fit(X_train_tsne, y_train)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=42, verbose=0,
                           warm_start=False)




```python
y_pred_tsne = rnd_clf_tsne.predict(X_test_tsne)
accuracy_score(y_test, y_pred_tsne)
```




    0.952



PCA로 차원축소된 데이터 학습 & 평가


```python
rnd_clf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf_pca.fit(X_train_pca, y_train)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=42, verbose=0,
                           warm_start=False)




```python
y_pred_pca = rnd_clf_pca.predict(X_test_pca)
accuracy_score(y_test, y_pred_pca)
```




    0.4965



지역적 선형 임베딩(LLE)로 차원축소된 데이터 학습 & 평가


```python
rnd_clf_lle = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf_lle.fit(X_train_lle, y_train)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=42, verbose=0,
                           warm_start=False)




```python
y_pred_lle = rnd_clf_lle.predict(X_test_lle)
accuracy_score(y_test, y_pred_lle)
```




    0.884



MDS로 차원축소된 데이터 학습 & 평가


```python
rnd_clf_mds = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf_mds.fit(X_train_mds, y_train_mds)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=42, verbose=0,
                           warm_start=False)




```python
y_pred_mds = rnd_clf_mds.predict(X_test_mds)
accuracy_score(y_test_mds, y_pred_mds)
```




    0.53



결과는 

*   tsne 사용 : 95.2%
*   pca기법 사용 : 49.65%
*   lle기법 사용 : 88.4%
*   mds기법 사용 : 53%

tsne를 사용하여 전처리를 진행했을때

RandomForestClassifier가 가장 분류를 잘하였다.


```python
import seaborn as sns

attr = ['TSNE','PAC','LLE','MDS']
v1 = [94.8, 39, 85.15, 47.75]
v2 = [95.2, 49.65, 88.4, 53]

plt.figure(figsize=(8,8)) ## Figure 생성 사이즈는 10 by 10
colors = sns.color_palette('hls',len(attr)) ## 색상 지정
xtick_label_position = list(range(len(attr))) ## x축 눈금 라벨이 표시될 x좌표
plt.xticks(xtick_label_position, attr) ## x축 눈금 라벨 출력
 
plt.bar(xtick_label_position, v1, color=colors) ## 바 차트 출력
plt.plot(attr, v2, c='black',label = '3D')
plt.title('Accuracy according to the kernel',fontsize=20) ## 타이틀 출력
plt.xlabel('Kernel') ## x축 라벨 출력
plt.legend()
plt.ylabel('Accuracy') ## y축 라벨 출력
plt.show()
```


    
![output_61_0](https://user-images.githubusercontent.com/42956142/121888610-2adc6a80-cd53-11eb-9b5a-b6485eba9d94.png)
    


그리고 2차원으로 차원축소 했을때보다

성능이 전체적으로 향상되었는데

이는 한 개의 차원(축)이 추가되어 그만큼 차원축소 데이터의 분산을

더 많이 가질수 있기 때문으로 생각된다.

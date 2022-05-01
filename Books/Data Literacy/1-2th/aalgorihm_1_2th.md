데이터 정제란 

1. 회귀
특정한 값을 예측함
a 자동차 크기 + b 자동차 무게 + c 라이트 세기d 문 갯수 = 가격

회귀에서 오차
MAE (mean absolute errer)

MSE (mean squared error)

RMSE (root mean squared error)



2. 분류
Logit 함수 - 분류의 Regression

우리 모델이 얼마나 퇴사 여부를 잘 예측 하여는가?

Accuracy : 정확도 (옳은 분류 갯수 / 전체 분류 갯수)

Precision : 정밀도 ( True Positve / )
        1명을 예측, 1명을 맞춤 100%     >>> 모델이 예측을 했을 때 그 값을 신뢰도 높게 예측
Recall : 재현율 ( True Postive / True Postive + False Negative)
        10명이 아픔, 1명만 아프다고 예측 >>> 10% 즉 모델을 얼마나 잘 찾아내는겨  
ROC Curve

AUC


3. 추천



# Algorith Selection

SVM
1. 클래스 사이에 
2. 알고리즘마다 결정에 해야할 파리미터가 있음
3. 케이스, 분류 마진


Decision Tree
1. 분류의 기준은 컴퓨터가 알아서 함
2. 비선형 데이터를 잘 처리함
3. 데이터가 작든 많든 덜 민간함
4. 오버피팅

Random Forest
1. 여러개의 트리를 만듬
2. 너무 오버피팅 하지않은 적절한 모델이 만들어짐

랜덤포레스트 vs 의사결정트리
1. 경우에 따라서 이 결과가 나왔냐고 설명을 해야하는데 이건 의사결정 트리가 더 설명하기 쉬움
2. 랜덤포레스트스는 수백가 트리가 있으니 설명하기 좀 어렵겠죠

인고지능 모델을 만드는 과정
1. 데이터를 모은다  >>>>> 점에 대해 뭔가를 한다 ,, 데이터가 갖고 있는 형태를 잘 주무른다
2. 데이터를 정제한다
3. 학습하고자 하는 인공지능 모델을 결정하낟
4. 해당 모델을 학습시킨다. >>>> 선을 잘그린다 ,, 모델의 성능을 고도화 시킨다
5. 모델을 평가한다.


데이터가 부족하 경우
- 오버 샘플링
- 

최적의 파라미터 찾기
 - 여러개 시도해보고, 그중에 가장 괜찮은거 생각해보기 


Random Sweep 
 - 모든 가능한 파라미터 다 있다고 생각해보자 // 랜덤 스윕은 이중에서 무작위로 골라 다해보고 그중에 가장 괜찮은걸로 가는거냐 당연히 많이 하면 할수록 좋은게 나올 확률이 올라가지
 - 
 -
Random Grid
- 선을 나눠서 무작위록, 간격을 어떻게 하냐에 따라 설정 

Entire Grid
 - 거의 모든 경우를 다해봄 ㅎㅎ


무작위로 하면 끝도없다 // 수행시간이 오래걸림 범위를 정해야 함 
 A B C D


''' 
# 이거 하면 랜덤 그리드로 가장 적절한 파라미터를 도출함, 그리고 도출한 결과는 모델임!! 
trial = 3



from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
import Orange.regression
import Orange.evaluation
import copy
import random
import numpy as np

data = copy.copy(in_data)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 5, stop = 200, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
#


best_model = None
best_MAE = None
best_parameter = None

for i in range(trial) : 
    __n_estimators = random.choice(n_estimators)
    __max_features = random.choice(max_features)
    __max_depth = random.choice(max_depth)
    __min_samples_split = random.choice(min_samples_split)
    __min_samples_leaf = random.choice(min_samples_leaf)
    __bootstrap = random.choice(bootstrap)
    
    print(__n_estimators, __max_features, __max_depth, __min_samples_split, __min_samples_leaf, __bootstrap)
    
    RF = Orange.regression.RandomForestRegressionLearner(n_estimators=__n_estimators,
                                                     max_features=__max_features,
                                                     max_depth=__max_depth,
                                                     min_samples_split=__min_samples_split,
                                                     min_samples_leaf=__min_samples_leaf,
                                                     bootstrap=__bootstrap)
    
    model = RF(data)
    cv_result = Orange.evaluation.testing.CrossValidation(data=data, learners=[RF])
    mae = Orange.evaluation.scoring.MAE(cv_result)

    print(mae)
    
    if best_MAE == None or best_MAE > mae[0] :
        best_MAE = mae[0]
        best_model = model
        best_parameter = {
            "n_estimators": __n_estimators,
            "max_features": __max_features,
            "max_depth": __max_depth,
            "min_samples_split": __min_samples_split,
            "min_samples_leaf": __min_samples_leaf,
            "bootstrap": __bootstrap
        }
        
print("BEST")

print(best_MAE)
print(best_parameter)

out_classifier=best_model

# 값을 전처리하는 방법
1. 값을 구간별로 그룹화 하기
2. Nomalization 
[0,1] , [-1,-1] 등 다양한 방법이 많음 알아서 찾아서 하자

# 피쳐 개수 정하기
1. 도메인 지식을 활용해서 파악하기
2. Score 점수를 활용해서 피처 개수 정하기 -- (Categorical , Numeric 즉 데이터 변수별로 사용하는 메소드가 달라짐)
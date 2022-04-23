import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model 

oecd_bil = pd.read_csv("oecd_bil_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_captia.csv", thousands=',', delimiter='\t', encoding='latin1', na_values="n/a")


'''
사이킷런의 설계 철학

사이킷런의 API는 아주 잘 설계되어 있습니다. 주요 설계 원칙은 다음과 같습니다.

일관성 : 모든 객체가 일관되고 단순한 인터페이스를 공유합니다.

- 추정기 : 데이터셋을 기반으로 일련의 모델 파라미터들을 추정하는 객체를 추정기라고 합니다 추장 자체는 fit() 메서드에 의해 숭되고
           하나의 매개변수로 하나의 데이터셋만 전달합니다( 지도 학습 알고리즘에서는 매개변수가 두 개로, 두 번째 데이터셋은 레이블을 담고 있습니다.)
           추정 과정에서 필요한 다른 매개변수들은 모두 하이퍼파리미터로 간주되고 인스터 변수로 저장됩니다.

- 변환기 : 
'''

# 교차 검증을 사용한 평가
'''
1. K-겹 교차 검증(K-Fold Cross-Validaiton)기능을 사용하는 방법이 있습니다.
다음 코드는 훈련 세트를 폴드라 불리는 10개의 서브셋으로 무작위로 분할합니다.
그런 다음 결정 트리 모델을 10번 훈련하고 평가하는데, 매번 다른 폴드를 선ㅌ갷 

''' 
from skelarn.model_selection import cross_val_socre
scores = cross_val_socre(tree_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error',cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_score(scores):
    print("점수", scores)
    print("평균", scores.mean())
    print("표쥰편차", socres.std())

import joblib

joblib.dump(my_model, 'my_model.pkl') #이거 데이터 모델을 사용하기
my_mode_loaded = joblib.load("my_model.pkl")

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

forest_rmse # 점수 도출
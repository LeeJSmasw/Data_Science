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

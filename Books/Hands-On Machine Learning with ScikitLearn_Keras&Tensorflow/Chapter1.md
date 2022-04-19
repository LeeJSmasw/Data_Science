
''' 
Utility function 효용 함수(또는 fitness function 적합도 함수)
비용함수 - cost funtion 
'''

1. 종류
 1.1 : 지도
  1.2 : 비지도
  1.3 : 배치 학습과 온라인 학습
  1.4 : 사례 기반 학습과 모델 깁나 학습

 -  머신러닝 프로젝트에서는 훈련 세트에 데이터를 모아 학습 알고리즘에 주입합니다. 학습 알고리즘이 모델 기반이면 훈련 세트에
 모델을 맞추기 위해 모델 파라미터를 조정하고(즉, 훈련 세트에서 좋은 예측을 만들기 위해), 새로운 데이터에서도 좋은 예측을 만들거라 기대합니다.

 2. 성능 
    2.1 RMSE


  - %matplotlib inline
    '''
    import matplotlib.pyplot as plt
    housing.hist(bins=50, figsize=(20,15))
    plt show 

  hist() 메서드는 맷플롯립을 사용하고 결국 화면에 그래프를 그리기 위해 사용자 컴퓨터의 그래픽 백엔드를 필요로 합니다.
  그래서 그래프를 그리기 전에 맷플롯립이 사용할 백엔드를 지정해줘야 합니다. 주피터의 매직 명령%matplotlib inline을 사용한다.

  이 명령은 맷플롭립이 주피터 자체의 백엔드를 사용하도록 설정합니다. 그러면 그래프는 노트북 안에 그려지게 됩니다.

  참고 사항 표준편차는 일반으적으로 시그마로 표시흔ㄴ데 이 값은 평균에서 떨어진 거리를 제곱하여 평균한 분산(variance)의 제곱근입니다. 어떤 특성이 정규분포 또는 가우시안 분포를 따르는 경우 68-95-99.7를 따르는 경우가 많습니다.
  1시그마 68%, 2시그마 95% 3시그마 99.7가 포함 ㅎㅎ 

  데이터 스누핑(Data Snooping, 편향) : 우리 뇌는 매우 과대적합기 되기 쉬운 패턴 감지 시스템입니다. 
  일부 패턴에 속아 특정 머신러닝 모델을 선택할 수 있습니다. 그러므로 성급한 일반화를 하지 않기 위해 조심해야 합니다.



from skelarn.compose import ColumnsTransformer

num_attrib = list(housing_num)

cat_attrib = ["ocean_proximity"]



full_pipeline = ColumnTransformer([

​	("num", num_pipeline, num_attribs),

​	("cat", OneHotEncoder(), cat_attribs),

])



housing_prepared = full_pipeline.fit_transform(housing)

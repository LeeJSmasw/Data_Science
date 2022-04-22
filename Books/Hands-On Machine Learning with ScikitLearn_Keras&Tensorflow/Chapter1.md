
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





''''

```python
mport pandas as pd
```

## movie data set은 여기서 다운로드 받으세요 URL https://datasets.imdbws.com/

### 시스템 환경은 MAC Mojave 입니다. 맥은 'Darwin'으로 표시됩니다.

```python
import platform
platform.system()
'Darwin'
```

#### '!'는 커맨드 명령어를 쓸수 있습니다.

```python
!pwd
/Users/kim/DACON Dropbox/GitHub/admin/tutorial3
```

#### .TSV 확장자는 '\' 포함된 채로 Read 됩니다. Reading 옵션을 넣어야 됩니다.

```python
pd.read_csv('/Users/kim/DACON Dropbox/Github/admin/data/IMDB/title.ratings.tsv').head()
```

|      | tconst averageRating numVotes |
| ---: | ----------------------------: |
|    0 |          tt0000001\t5.7\t1528 |
|    1 |           tt0000002\t6.3\t186 |
|    2 |          tt0000003\t6.6\t1173 |
|    3 |           tt0000004\t6.3\t114 |
|    4 |          tt0000005\t6.2\t1889 |

#### read_csv( ' 경로 / 파일 ' , ??? ) 내에 sep = ' \ t ' 를 넣어 주시면 됩니다.

```python
movie = pd.read_csv('/Users/kim/DACON Dropbox/Github/admin/data/IMDB/title.ratings.tsv', sep='\t')
movie.head()
```

|      |    tconst | averageRating | numVotes |
| ---: | --------: | ------------: | -------: |
|    0 | tt0000001 |           5.7 |     1528 |
|    1 | tt0000002 |           6.3 |      186 |
|    2 | tt0000003 |           6.6 |     1173 |
|    3 | tt0000004 |           6.3 |      114 |
|    4 | tt0000005 |           6.2 |     1889 |

### 타임시리즈 Sorting 과 판다스 Sorting 을 배워봅니다.

### 1. pandas.core.series.Series

- Acending 오름차순이 기본으로 설정되어 있습니다.

```python
movie.numVotes.sort_values().head()
812809    5
66725     5
782808    5
514583    5
782809    5
Name: numVotes, dtype: int64
```

- Ascending = False 내림차순으로 변경합니다.

```python
movie.numVotes.sort_values(ascending=False).head()
80338     2118893
239741    2084449
499345    1857734
96407     1693706
80122     1658087
Name: numVotes, dtype: int64
```

- 시리즈 Type Check

```python
type(movie.numVotes.sort_values(ascending=False).head())
pandas.core.series.Series
```

### 2. 판다스 sorting 정렬

```python
movie.sort_values('numVotes').head()
```

|        |    tconst | averageRating | numVotes |
| -----: | --------: | ------------: | -------: |
| 812809 | tt5478670 |           8.4 |        5 |
|  66725 | tt0094798 |           5.6 |        5 |
| 782808 | tt4859704 |           4.2 |        5 |
| 514583 | tt1467291 |           9.4 |        5 |
| 782809 | tt4859730 |           7.8 |        5 |

- 데이터프레임 Type Check

```python
type(movie.sort_values('numVotes').head())
pandas.core.frame.DataFrame
```

- Ascending = False

- Tip1: movie.sort_values( 문자 3개를 입력 + 탭 ) 자동완성 됩니다
- Tip2: movie.sort_values( 쉬프트 + 탭 ) => 해당 메서스의 옵션들을 볼 수 있습니다.

```python
movie.sort_values('numVotes', ascending=False).head()
```

|        |    tconst | averageRating | numVotes |
| -----: | --------: | ------------: | -------: |
|  80338 | tt0111161 |           9.3 |  2118893 |
| 239741 | tt0468569 |           9.0 |  2084449 |
| 499345 | tt1375666 |           8.8 |  1857734 |
|  96407 | tt0137523 |           8.8 |  1693706 |
|  80122 | tt0110912 |           8.9 |  1658087 |

- 두개의 컬럼도 소팅이 가능 합니다.

```python
movie.sort_values(['numVotes', 'averageRating'], ascending=False).head()
```

|        |    tconst | averageRating | numVotes |
| -----: | --------: | ------------: | -------: |
|  80338 | tt0111161 |           9.3 |  2118893 |
| 239741 | tt0468569 |           9.0 |  2084449 |
| 499345 | tt1375666 |           8.8 |  1857734 |
|  96407 | tt0137523 |           8.8 |  1693706 |
|  80122 | tt0110912 |           8.9 |  1658087 |

- 컬럼이 순서가 바뀌면 소팅이 달라집니다. 아래의 순서를 보세요

```spython
movie.sort_values(['averageRating','numVotes' ], ascending=False).head()
```

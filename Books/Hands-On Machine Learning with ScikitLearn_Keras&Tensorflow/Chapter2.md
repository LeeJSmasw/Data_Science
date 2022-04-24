```python
import pandas as pd
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



housing.plot(kind = 'scatter', x = 'median_income', y = 'median_house_value', alpha=0.1)



특성 조합으로 실험

데이터를 탐색하고 통찰을 얻는 여러 방법에 대한 아이디어를 얻었기 바랍니다.

머신러닝 알고리즘에 주입하기 전에 정제해야 할 조금 이상한 데이터를 확인했고, 특성 사이(특히 타깃 속성과 사이)에서 흥미로운 상관관계를 발견할 수 있습니다.



1. 로그 스케일

2. 여러 특성의 조합을 시도 ( 특정 방 개수는 얼마나 많은 가구 수가 있는지, 방 보단 가구당 방 개수 등 )

   housing['rooms_per_household'] = housing['total_rooms'] / housing[households]

   housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']

   housing['population_per_household'] = housing['population/'] / housing['households']

--- 이런식으로 





2.5 머신러닝 알고리즘을 위한 데이터 준비

1. 어떤 데이터셋에 대해서도 데이터 변환을 손쉽게 반복할 수 있어야 함
2. 향후프로젝트에 사용할 수 있는 변환 라이브러리를 점진적으로 구축하게 됩니다.



먼저 원래 훈련 세트로 복원하고, 예측 변수와 타겟값에 같은 변형을 적용하지 않귀 위해 예측 변수와 레이브를 분리합니다.(drop())데이터 복사본을 만들며 starts_train_set에 영향을 주지 않니당



2.5.1 데이터 정제

1. 해당 구역을 제거
2.  전체 특성을 제거
3. 어떤 값을 채움



dropna (), drop(), filna() 메서드 등 



median = housing['total_bedrooms'].medain()

housing['total_bedrooms'].filna(medain, inpalce=True)



from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')



housing_num = housing.drop('ocean_proximity',axis=1)



imputer.fit('ocean_proximity', axis=1)

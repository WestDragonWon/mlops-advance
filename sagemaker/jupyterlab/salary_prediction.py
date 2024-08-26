# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
# !pip install tensorflow
# !pip install seaborn

# %%
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# %%
# read the csv file 
df = pd.read_csv('salary.csv')
df

# %%
# 1. 데이터프레임의 처음 7개 행과 마지막 7개 행을 출력
print("처음 7개 행:")
print(df.head(7))
print("\n마지막 7개 행:")
print(df.tail(7))

# %%
# 2. 최대 급여 값을 찾기
max_salary = df['Salary'].max()
print(f"\n최대 급여 값: {max_salary}")

# 3. 연차별 평균 급여 계산
avg_salary_per_year = df.groupby('YearsExperience')['Salary'].mean()
print("\n연차별 평균 급여:")
print(avg_salary_per_year)

# 4. 연차와 급여 사이의 상관관계 계산
correlation = df['YearsExperience'].corr(salary_df['Salary'])
print(f"\n연차와 급여 사이의 상관계수: {correlation}")

# 5. 급여가 100,000 이상인 사람들의 연차 확인
high_salary_experience = df[salary_df['Salary'] >= 100000]['YearsExperience']
print("\n급여가 100,000 이상인 사람들의 연차:")
print(high_salary_experience)

# %%
# check if there are any Null values
df.isna().sum()

# %%
df.groupby('YearsExperience').mean()

# %%
df.corr()

# %%
df[df['Salary'] >= 100000]

# %%
df.info()


# %%
df.describe()

# %% [markdown]
# 급여가 가장 적은 직원과 가장 많은 직원의 연차(근무 경험 연수)는 각각 몇 년입니까?

# %%
# 최저 급여와 최고 급여 값 찾기
min_salary = df['Salary'].min()
max_salary = df['Salary'].max()

# 최저 급여와 최고 급여에 해당하는 연차 찾기
min_salary_experience = df[df['Salary'] == min_salary]['YearsExperience'].values[0]
max_salary_experience = df[df['Salary'] == max_salary]['YearsExperience'].values[0]

print(f"최저 급여에 해당하는 연차: {min_salary_experience}년")
print(f"최고 급여에 해당하는 연차: {max_salary_experience}년")

# %%
df.hist(bins = 30, figsize = (20,10), color = 'r')

# %%
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot = True)
plt.show()

# %%
sns.pairplot(df)

# %%
# 회귀선
sns.regplot(data=df, x='YearsExperience', y='Salary')

# %%
sns.regplot(x='YearsExperience', y='Salary', data=df)

# 그래프 제목 및 레이블 추가
plt.title('Salary vs. Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

# 그래프 출력
plt.show()

# %%
# 회귀방정식 => 연차를 입력했을 때 기대되는 샐러리 예측(단순회귀)
# - 생산관리팀: 1달에 제품이 몇개 나와야하는데 20일지나서 지금 몇개야 속도 괜찮나?
# - 재료 언제 구매해야해?

# (1) 모델(LinearRegression) (2) numpy로 회귀방정식 (기울기, 절편)
from sklearn.linear_model import LinearRegression

X = df[['YearsExperience']]
y = df[['Salary']]

model = LinearRegression()
model.fit(X, y)

# %%
# 회귀계수
beta1 = model.coef_[0] #기울기
beta0 = model.intercept_ # 절편

# %%
y = beta1 * 13.5 + beta0
y

# %%
# numpy를 활용한 방법

X = df['YearsExperience'] # 1차원 Series / 2차원 DataFrame
y = df['Salary']

beta1, bata0 = np.polyfit(X, y, 1)
beta1, beta0

# %%
x = 10
y = 8731.941047062493 * x + 8731.941047062493
y

# %% [markdown]
# # Model 학습

# %%
# 학습 데이터와 테스트 데이터 분리
from sklearn.model_selection import train_test_split

X = df[['YearsExperience']]
y = df[['Salary']]

X = np.array(X).astype(np.float32)
y = np.array(y).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

X_train.shape, X_test.shape

# %%
linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

# %%
accuracy = linear_model.score(X_test, y_test)

accuracy

# %%
linear_model.predict(X_test)

# %% [markdown]
# ## SageMaker를 활용한 모델 최적화 과정

# %%
# DE

# 스타트업 개발자라는 직군이 직업 만족도가 높다
# 만들고 싶은 걸 만들어야 좋다.
# De -> 데이터 분산 저장. Queue -> Worker

# %%
from sagemaker import Session
import sagemaker
import boto3

# 항상연결하면 돈나감.. 필요할때만 연결
session = sagemaker.Session()
bucket = session.default_bucket()
prefix = 'employee-salary-linear'
role = sagemaker.get_execution_role()
role

# %%
import io
import sagemaker.amazon.common as smac

X = df[['YearsExperience']]
y = df[['Salary']]

X = np.array(X).astype(np.float32)
y = np.array(y).astype(np.float32)

X_train = X_train if isinstance(X_train, np.ndarray) else X_train.values
y_train = y_train.ravel() if isinstance(y_train, np.ndarray) else y_train.values.ravel()

buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, X_train, y_train)
buf.seek(0)

# %%
# Code, Data -> S#
import os
key = 'linear-train-data'

boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)

s3_train_data = f"s3://{bucket}/{prefix}/train/{key}"
print(f"S3에 업로드된 학습 데이터: {s3_train_data}")

# %%
X_train = X_train if isinstance(X_train, np.ndarray) else X_train.values
y_train = y_train.ravel() if isinstance(y_train, np.ndarray) else y_train.values.ravel()

buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, X_train, y_train)
buf.seek(0)

# %%
# s3 모델의 출력 결과물을 저장할 위치 지정
output_location = f's3://{bucket}/{prefix}/output'

# %%
# SageMaker 내장되어있는 모델을 불러와서 사용

import boto3
from sagemaker import image_uris

container = image_uris.retrieve('linear-learner', boto3.Session().region_name)


# %%
# 데이터를 float32로 변환
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)

# 변환된 데이터를 S3에 업로드
buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, X_train, y_train)
buf.seek(0)  # 버퍼의 처음으로 이동하여 데이터를 업로드할 준비

key = 'linear-train-data'
boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)  # 이 시점에서 buf는 닫히지 않았음

print(f"S3에 업로드된 학습 데이터: {s3_train_data}")

# SageMaker 내장된 모델을 사용하여 학습
linear = sagemaker.estimator.Estimator(
    container,
    role,
    instance_count=1,
    instance_type='ml.c4.xlarge',
    output_path=output_location,
    sagemaker_session=session
)

linear.set_hyperparameters(
    feature_dim=1,
    predictor_type='regressor',
    mini_batch_size=5,
    epochs=5,
    num_models=32,
    loss='squared_loss'
)

linear.fit({'train': s3_train_data})

# %%
#모델생성 엔드포인트 구성 및 배포
linear.deploy(
    initial_instance_count=1,  
    instance_type='ml.m4.xlarge'
)

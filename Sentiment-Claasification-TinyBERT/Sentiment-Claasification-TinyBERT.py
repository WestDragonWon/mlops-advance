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
#     display_name: mlops-basic
#     language: python
#     name: python3
# ---

# %%
## 가상환경
# !pip install pandas datasets

# %%
## 데이터 로드

import pandas as pd

data = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/IMDB-Dataset.csv')

data.head()


# %%
from datasets import Dataset
dataset = Dataset.from_pandas(data)
dataset = dataset.train_test_split(test_size=0.3)
dataset

# %%
label2id = {'negative': 0, 'positive': 1}

dataset = dataset.map(lambda x: {'label': label2id[x['sentiment']]})

# %%
# 데이터셋을 저장 -> Tokenization(텍스트를 모델이 이해할 수 있는 숫자 형식으로 변환)
# !pip install transformers torch

# %%
from transformers import AutoTokenizer
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Hugging Face의 사전 훈련된 모델을 사용. TinyBERT 모델을 사용
model = 'huawei-noah/TinyBERT_General_4L_312D'

tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
tokenizer

# %%
tokenizer(dataset['train'][0]['review'])

# %%
tokenizer('Today is monday')


# %%
def tokenize(batch):
    temp = tokenizer(batch['review'], padding=True, truncation=True, max_length=300)
    return temp

dataset = dataset.map(tokenize, batched=True, batch_size=None)


# %%
dataset['train'][0]

# %%
# Model Build
# https://huggingface.co/docs/transformers/v4.42.0/en/tasks/sequence_classification#evaluate
# !pip install evaluate accelerate scikit-learn

# %%
import evaluate
import numpy as np

accuracy = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1) # 예측값과 실제 레이블 데이터를 튜플로 입력

    return accuracy.compute(predictions=predictions, references=labels)


# %%
# !pip install tf-keras

# %%
from transformers import AutoModelForSequenceClassification

label2id = { 'negative': 0, 'positive': 1 }
id2label = {0: 'negative', 1: 'positive'}

model = AutoModelForSequenceClassification.from_pretrained(
    model,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)

# %%
from transformers import TrainingArguments, Trainer

# 모델 학습을 위한 하이퍼파라미터와 설정 정의
args = TrainingArguments(
    output_dir='train_dir',               # 학습 결과를 저장할 디렉터리
    overwrite_output_dir=True,            # 출력 디렉터리에 이미 있는 파일을 덮어쓸지 여부
    num_train_epochs=3,                   # 학습할 에포크(epoch) 수
    learning_rate=2e-5,                   # 학습률 (learning rate)
    per_device_train_batch_size=32,       # 각 디바이스(예: GPU)당 학습 배치 크기
    per_device_eval_batch_size=32,        # 각 디바이스당 평가 배치 크기
    evaluation_strategy='epoch'           # 평가 전략 (여기서는 매 에포크마다 평가)
)

# Trainer 객체를 생성하여 학습 및 평가를 관리
trainer = Trainer(
    model=model,                          # 학습할 모델
    args=args,                            # 학습 파라미터 설정
    train_dataset=dataset['train'],       # 학습에 사용할 데이터셋
    eval_dataset=dataset['test'],         # 평가에 사용할 데이터셋
    compute_metrics=compute_metrics,      # 평가 지표를 계산하는 함수
    tokenizer=tokenizer                   # 토크나이저 (텍스트를 토큰으로 변환하는 도구)
)

# %%
trainer.train()

# %%
trainer.evaluate()

# %%
trainer.save_model('tinybert-sentiment-analysis')

# %%
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1

classifier = pipeline('text-classification', model='tinybert-sentiment-analysis', device=device)

# %%
data = [
    "The Chinese WW2 spy thriller “Decoded” stands out for a number of reasons, mostly in spite of its conventional and hackneyed depiction of a troubled mathematician who deciphers encrypted messages for the mainland army. For starters, “Decoded” provides a dramatic change of pace for two marquee-worthy names: soft-spoken heart-throb Liu Haoran, who takes an unusual leading man role as the gifted, but painfully shy codebreaker Rong Jinzhen; and director Chen Sicheng, who’s best known for his goofy mega-blockbuster “Detective Chinatown” comedies. With “Decoded,” a plodding adaptation of Mai Jia’s popular source novel, Chen and Liu abandon cheap-seats humor—Liu co-starred in the “Detective Chinatown” movies, playing a straight man to comedian Wang Baoqiang—to pursue a more sober, but less convincing type of cornball power fantasy.",
    "Liu also played a frustrated, but superhumanly gifted wallflower in “Detective Chinatown.” He was more convincing in those movies, partly because he was part of a winning buddy duo, but also because he wasn’t trying to capital-A act while wearing hairpieces, whose synthetic hairs thin at an alarming rate as his character ages. As Jinzhen, Liu brings to mind Russell Crowe’s performance as the schizophrenic mathematician John Nash in “A Beautiful Mind.” That association gets harder and harder to shake as Jinzhen inevitably loses his grip on reality while trying to solve the Black Cipher, a nigh-impossible encryption key that was specifically designed to stump Jinzhen.",
    "Liu’s mostly compelling as a leading man whenever he can suggest a lot about Jinzhen by speaking softly and deferring his gaze, as if Jinzhen expects to be reprimanded or inconvenienced at any time. He’s still often eclipsed by co-star John Cusack, whose broad and twitchy performance often distracts from his dialogue, as well as a series of campy dream sequences that ostensibly speak for Liu’s introverted protagonist.",
    "Liu doesn’t exactly light up the screen in his limited capacity as a humanoid plot device. His character either reacts to or follows after whatever promising new development might help Jinzhen to solve the latest problem that’s vexing him. The filmmakers do what they can to compensate for their unlikely hero’s prevailing lack of charm and agency, but not even the combined forces of Lloyd Dobler and the Fab Four can bring a spike of joy to this DOA period drama."
]

classifier(data)

# %%
data=[
    """When are you guys going to fix all the issues?? Firstly, none of the reaction emojis are showing up. It's just a grey circle. When scrolling, it doesn't move freely. There's like a delay!! Very frustrating!!! Also, nearly every post is either from a "suggested page" or a "sponsered page". I hardly ever see anything from the pages that I actually follow or my friends pages. No wonder so many people are leaving FB 🙄🙄"""
]

classifier(data)

# 프로그램: 구글 플레이 스토어 링크를 넣으면 => 리뷰 데이터 전체 크롤링 => 부정의 강도가 0.8 이상인 의견만 필터 걸어서 고객사에게 공유
# 별1개에 네거티브 0.8이상 => slack 으로 알림보내 => 대응

# %%
# 모델을 s3에 업로드
# (1) AWS 로그인 한 다음 => 버킷 생성
# (2) boto3를 활용해서 코드 베이스 s3 생성 및 파일 업로드
# !pip install boto3

# %%
import boto3

s3 = boto3.client('s3') # s3 만들 수 있는 권한 / 계정(aws credentials)

# s3.list_buckets()
# s3.create_bucket()

# 어떤 원리로 실행이 되는 걸까요?

# aws credentials
# aws configure list

# %% [markdown]
# (1) IAM 계정 생성
# (2) AWS CLI 명령어 설치
# https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
#
# > aws version
# > aws configure
# ```
# 1. AW Acc k : [YourAWSAccessKeyHere]
# 2. AW Secret acc K :[YourAWSSecretKeyHere]
# Default region name :  [YourAWSregionHere]
# Default output format [json]: json
# ```
# > aws confiure list

# %%
# S3 버킷 생성
import boto3
import time
from botocore.exceptions import ClientError

s3 = boto3.client('s3') # s3 콘솔에 접속
bucket_name = 'inseop-mlmodels' # bucket name

def create_bucket(bucket_name):
    response = s3.list_buckets()

    bucket_list = []
    for buck in response['Buckets']:
        bucket_list.append(buck['Name'])

    if bucket_name not in bucket_list:
        try:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint':'ap-northeast-2'}
            )
        except ClientError as e:
            print('오류 발생 :', e)

            if e.response['Error']['Code'] == 'BucketAlreadyExists':    
                print('다른 버킷 이름을 입력하세요.')

            elif e.reponse['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                print('이미 만들어 져있는 버킷입니다.')
                
            else:
                print('버킷 만들기를 재시도 중입니다.')
                time.sleep(3)
                create_bucket(bucket_name)


# %%
create_bucket('inseop-mlmodelss')

# %%
# DE => 데이터가 유실되었을 때 어떻게 복구할 것인가
# AB180 => 맨날 복구. 더미 데이터에서 데이터 불러와서 => 다시 재가공 => 다시 DB insert
# 하루, 한시간

# %% [markdown]
# S3에 파일(폴더) 업로드

# %%
import boto3

s3 = boto3.client('s3')
bucket_name = 'inseop-mlmodels'

s3.upload_file

# %%
import os

for root, dir, files in os.walk('tinybert-sentiment-analysis'):
    # print(root, dir, files)

    for file_name in files:
        # print(file_name)

        file_path = os.path.join(root, file_name)
        print(file_path)
    
        # s3.upload_file(file_path, bucket_name, file_name)
        s3.upload_file(file_path, bucket_name, file_path)

# %%
import os

def s3_upload_file_folder_name(model_folder, folder_name):
    for root, dir, files in os.walk(model_folder):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            s3_key = os.path.join(folder_name, file_name)
            s3.upload_file(file_path, bucket_name, s3_key)
            # EC2에다가 docker => EC2 - AWS CLI(credentials은 필요X) + Docker pull


# %%
s3_upload_file_folder_name('tinybert-sentiment-analysis', 'tinybert-test-folder')

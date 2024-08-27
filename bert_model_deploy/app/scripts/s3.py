import boto3
import os

s3 = boto3.client('s3')
paginator = s3.get_paginator('list_objects_v2')
bucket_name = 's3-bucket-name'

def download_dir(local_path, model_name):
    if not os.path.exists(local_path):
        print('Creating directory:', local_path)
        os.makedirs(local_path)

    for result in paginator.paginate(Bucket='bucket_name', Prefix='path/to/folder'):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key'] # tinybert-sentiment-analysis

                local_file = os.path.join(local_path, os.path.relpath(s3_key, 'tinybert-sentiment-analysis'))
                print('local_file:', local_file)

                # 상위 폴더 생성
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                
                s3.download_file('bucket_name', s3_key, local_file)
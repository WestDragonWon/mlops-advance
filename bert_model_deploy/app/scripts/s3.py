import boto3
import os

s3 = boto3.client('s3')
paginator = s3.get_paginator('list_objects_v2')
bucket_name = 's3-bucket-name'

def download_dir(local_path, model_name):
    for result in paginator.paginate(Bucket='bucket_name', Prefix='path/to/folder'):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key'] # tinybert-sentiment-analysis

                local_file = os.path.join(local_path, os.path.relpath(s3_key, 'tinybert-sentiment-analysis'))
                s3.download_file('bucket_name', s3_key, local_file)
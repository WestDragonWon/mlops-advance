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

# %% [markdown]
# ## EC2 관리
#
# keypair 생성 cli
# https://docs.aws.amazon.com/cli/latest/reference/ec2/create-key-pair.html
#
# aws ec2 create-key-pair --key-name fastapi-backend-keypair --query 'KeyMaterial' --output te
# xt > fastapi-backend-keypair.pem

# %%
import boto3

# s3 = boto3.client('s3')
ec2 = boto3.client('ec2', region_name='ap-northeast-2')

instance_name = 'FastAPI-Sentiment-BERT-Backend' # git repo

# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/run_instances.html
#  
response = ec2.run_instances(
    ImageId='ami-00a08b445dc0ab8c1',
    MinCount=1,
    MaxCount=1,
    InstanceType='t2.micro',
    KeyName='fastapi-backend-keypair',
    BlockDeviceMappings=[             # 블록 디바이스 설정 (스토리지)
        {
            "DeviceName": "/dev/xvda", # 기본 디바이스 이름
            'Ebs':{
                'DeleteOnTermination': True, # 인스턴스 종료 시 볼륨 삭제 여부
                'VolumeSize': 120            # 루트 볼륨 크기 (GB)
            }
        }
    ]   
)

# %%
instance_id = response['Instances'][0]['InstanceId']

ec2.create_tags(
    Resources=[instance_id],
    Tags=[{"Key":"Name", "Value":instance_name}]
)

# %%
# 보안그룹설정
group_name = 'fastapi-backend-security-group'

response = ec2.describe_security_groups()
group_id = [x['GroupId'] for x in response['SecurityGroups'] if x['GroupName']== group_name ]

if group_id == []:
    response = ec2.create_security_group(
        Description='fastapi backend security group for bert model deploy', 
        GroupName=group_name,
    )
    group_id = response['GroupId']
else:
    group_id = group_id[0]

group_id


# %%
# 인바운드/아웃바운드 => 열어준다음에
# EC2에 도커 컨테이너 얹으면 끝.

def security_group_add_inbound(group_id, protocol, port, cidr):
    reponse = ec2.authorize_security_group_ingress(
        GroupId=group_id,
        IpPermissions=[
            {
                'IpProtocol': protocol,
                'FromPort': port,
                'ToPort': port,
                'IpRanges': [{'CidrIp': cidr}]
            }
        ]   
    )

security_group_add_inbound(group_id, 'tcp', 22, '0.0.0.0/0') # SSH # 생산성 # 스타벅스
security_group_add_inbound(group_id, 'tcp', 80, '0.0.0.0/0') # 일반 호스트 서버 접속

security_group_add_inbound(group_id, 'tcp', 8501, '0.0.0.0/0') # Streamlit
security_group_add_inbound(group_id, 'tcp', 8502, '0.0.0.0/0')

# %%
# 방금 만든 Security Group을 EC2에 적용합니다.

ec2.modify_instance_attribute(
    InstanceId=instance_id,
    Groups=[group_id]
)

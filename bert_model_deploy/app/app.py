from fastapi import FastAPI
from scripts.data_model import NLPDataInput, NLPDataOutput

# model loader
import torch
import os
import uvicorn
from scripts.s3 import download_dir
from transformers import pipeline

app = FastAPI()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_name = 'tinybert-sentiment-analysis/'
local_path = 'ml-models/'+model_name #S3 경로

if not os.path.isdir(local_path):
    download_dir(local_path, model_name)

sentiment_model = pipeline('text-classification', model=local_path, device=device)

@app.get
def root():
    return {"hello":"FastAPI"}

import time
@app.post('/api/v1/sentiment') # 127.0.0.1:8000/api/v1/sentiment 에서 POST 요청을 받음
def sentiment_analysis(data: NLPDataInput): # payload: 프론트에서 보내주는 데이터
    start_time = time.time()

    output = sentiment_model(data.text)

    labels = [x['label'] for x in output]
    scores = [x['score'] for x in output]

    # labels = []
    # for x in output:
    #     labels.append(x['label'])

    end_time = time.time()
    prediction_time = int((end_time - start_time)*1000)


    result = NLPDataOutput(
        model_name = 'tinybert-sentiment-analysis',
        text = data.text,
        labels = labels,
        scores = scores,
        prediction_time = prediction_time,
    )

    return result

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app="app:app", port=8000, reload=True)
                


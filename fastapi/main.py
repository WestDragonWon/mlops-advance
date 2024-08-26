from fastapi import FastAPI
from PIL import Image
from io import BytesIO
from fastapi import UploadFile, File
from predict import predict

app = FastAPI()

# 라우팅 경로를 설정
@app.get('/') # 127.0.0.1:8000/ 경로로 요청이 들어오면
def root():
    return {"hello":"FastAPI"}

@app.get('/items/{item_id}')
def shwo_item(item_id):
    return {"hello":item_id}

@app.post('/api/v1/predict')
async def image_predict_api(file: UploadFile = File(...)):
    raw_data = await file.read()
    image_bytes_io = BytesIO(raw_data)
    print(image_bytes_io)
    image = Image.Open(image_bytes_io)
    pred = predict

    return pred


# uvicorn main:app 
# FastAPI 가장 큰 특징 2개
# - 비동기 -> python 은 동기인데 ASGI 서버를 사용해서 비동기로 동작
# WSGI 서버는 동기로 동작
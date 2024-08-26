from pydantic import BaseModel

# FastAPI에서 REST API를 구현할 때와 같음
class NLPDataInput(BaseModel):
    text: list[str]
    username: str

class NLPDataOutput(BaseModel):
    Model_name: str
    text: list[str]
    labels: list[str]
    scores: list[float]
    prediction_time: int
    


from fastapi import FastAPI
from pydantic import BaseModel

import senti 

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str


app = FastAPI()

@app.post("/predict")
def predict(request: SentimentRequest) -> SentimentResponse:
    sentiment = senti.predict_sentiment(request.text)
    print(f"The model returned the sentiment: {sentiment}")
    return SentimentResponse(sentiment=sentiment) 
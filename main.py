from fastapi import FastAPI
from pydantic import BaseModel
import model.model
import json
import uvicorn
from pyngrok import ngrok
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SentimentRequest(BaseModel):
    text: list


class SentimentResponse(BaseModel):
    label: list
    #probability: list


@app.get("/")
def home():
    return {"health_check": "OK"}


@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest):
    sentiment = model.model.predict(request.text) #, probability
    return SentimentResponse(label=sentiment) #, probability=probability

ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)
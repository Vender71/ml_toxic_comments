from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import torch
import time

from sys import platform
import uvicorn


class Message(BaseModel):
    text: str
    mode: str

# Преобразование выходных логарифмов модели в оценку вероятности.       
  def softmax(x):
      e_x = np.exp(x - np.max(x))
      return e_x / e_x.sum()

# Функция принимает сообщение в качестве входных данных и на основе BERT возвращает степень токсичности
  def handler_message(message):
      if message.mode in ["all", "neutral", "toxic"]:
          mode = message.mode
      else:
        mode = "all"

    batch = tokenizer.encode(message.text, return_tensors="pt")
    start_time = time.time()
    with torch.no_grad():
        outputs = model(batch)
        outputs = outputs.logits
    end_time = time.time()
    predictions = softmax(outputs.cpu().detach().numpy())
    predictions = predictions.flatten()

    result = {"text": message.text}

    if mode in ["all", "neutral"]:
        result["neutral"] = "{:.2f}".format(predictions[0])
    if mode in ["all", "toxic"]:
        result["toxic"] = "{:.2f}".format(predictions[1])

    return result


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# load tokenizer and model weights
tokenizer = BertTokenizer.from_pretrained(
    "SkolkovoInstitute/russian_toxicity_classifier"
)
model = BertForSequenceClassification.from_pretrained(
    "SkolkovoInstitute/russian_toxicity_classifier"
)

# Функция словарь с сообщением, указывающий на работоспособность.
@app.get("/")
def root():
    return {"message": "Greats! It's work!"}

@app.get("/check/message/{text}/")
def check_message(text):
    message = Message(text=text, mode="all")
    return handler_message(message)

# Эта функция принимает одно входное сообщение для классификации токсичности сообщения.
@app.post("/check/message/")
def check_message(message: Message):
    return handler_message(message)

# Функция  обрабатывает сообщение, вызывая функцию handler_message и возвращает результат для каждого из них.
@app.post("/check/messages/")
def check_message(messages: List[Message]):
    message_results  = list()
    for message in messages:
        message_results .append(handler_message(message))

    return message_results 


# run worker
if __name__ == "__main__":
    uvicorn.run(
        app,
        port=5049 if platform == "win32" else 8000,
        host="127.0.0.1" if platform == "win32" else "0.0.0.0",
        workers=1,
        log_level="info",
    )

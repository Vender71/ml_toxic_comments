from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import torch
from sys import platform
import uvicorn


class Message(BaseModel):
    text: str
    mode: str


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def handler_message(message):
    if message.mode in ["all", "neutral", "toxic"]:
        mode = message.mode
    else:
        mode = "all"

    batch = tokenizer.encode(message.text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(batch)
        outputs = outputs.logits
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

tokenizer = BertTokenizer.from_pretrained(
    "SkolkovoInstitute/russian_toxicity_classifier"
)
model = BertForSequenceClassification.from_pretrained(
    "SkolkovoInstitute/russian_toxicity_classifier"
)


@app.get("/")
def root():
    return {"message": "Greats! It's work!"}


@app.get("/check/message/{text}/")
def get_check_message(text):
    message = Message(text=text, mode="all")
    return handler_message(message)


@app.post("/check/message/")
def post_check_message(message: Message):
    return handler_message(message)


@app.post("/check/messages/")
def post_check_messages(messages: List[Message]):
    message_results = list()
    for message in messages:
        message_results.append(handler_message(message))

    return message_results


if __name__ == "__main__":
    uvicorn.run(
        app,
        port=5049 if platform == "win32" else 8000,
        host="127.0.0.1" if platform == "win32" else "0.0.0.0",
        workers=1,
        log_level="info",
    )

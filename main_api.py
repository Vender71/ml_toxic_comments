from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import numpy as np
import torch


class Message(BaseModel):
    text: str
    mode: str

def softmax(x):
   e_x = np.exp(x - np.max(x))
   return e_x/e_x.sum()

def handler_message(message):
    if message.mode in ['all', 'neutral', 'toxic']:
        mode = message.mode
    else:
        mode = 'all'
    
    batch = tokenizer.encode(message.text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(batch)
        outputs = outputs.logits   
    predictions = softmax(outputs.cpu().detach().numpy())
    predictions = predictions.flatten()
    
    result = {
            'text': message.text
        }
    
    if mode in ['all', 'neutral']:
        result['neutral'] = '{:.2f}'.format(predictions[0])
    if mode in ['all', 'toxic']:
        result['toxic'] = '{:.2f}'.format(predictions[1])
        
    return result

app = FastAPI()
# load tokenizer and model weights
tokenizer = BertTokenizer.from_pretrained('SkolkovoInstitute/russian_toxicity_classifier')
model = BertForSequenceClassification.from_pretrained('SkolkovoInstitute/russian_toxicity_classifier')

@app.get("/")
def root():
    return {"message": "Greats! It's work!"}

@app.post("/check/message/")
def check_message(message: Message):
    return handler_message(message)

@app.post("/check/messages/")
def check_message(messages: List[Message]):
    res_arr = list()
    for message in messages.text:
        res_arr.append(handler_message(message))
        
    return res_arr
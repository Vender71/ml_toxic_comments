from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import torch


def softmax(x):
   e_x = np.exp(x - np.max(x))
   return e_x/e_x.sum()


# load tokenizer and model weights
tokenizer = BertTokenizer.from_pretrained('SkolkovoInstitute/russian_toxicity_classifier')
model = BertForSequenceClassification.from_pretrained('SkolkovoInstitute/russian_toxicity_classifier')

# prepare the input
batch = tokenizer.encode('ты супер', return_tensors='pt')

# inference
with torch.no_grad():
    outputs = model(batch)
    outputs = outputs.logits

predictions = softmax(outputs.cpu().detach().numpy())
predictions = predictions.flatten()
print('neutral {:.2f} toxic {:.2f}'.format(predictions[0], predictions[1]))

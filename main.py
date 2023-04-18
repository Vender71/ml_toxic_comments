from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import torch
import streamlit as st


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# upload tokenizer
@st.cache(allow_output_mutation=True)
def load_tokenizer():
    return BertTokenizer.from_pretrained(
        "SkolkovoInstitute/russian_toxicity_classifier"
    )


# uload model
@st.cache(allow_output_mutation=True)
def load_model():
    return BertForSequenceClassification.from_pretrained(
        "SkolkovoInstitute/russian_toxicity_classifier"
    )


st.title("Классификация токсичности комментария")

# prepare the input
comment = st.text_input(label="Введите кoмментарий", value="")

model = load_model()

tokenizer = load_tokenizer()

batch = tokenizer.encode(comment, return_tensors="pt")

result = st.button("Классифицировать комментарий")

# inference
if result:
    with torch.no_grad():
        outputs = model(batch)
        outputs = outputs.logits
    predictions = softmax(outputs.cpu().detach().numpy())
    predictions = predictions.flatten()
    st.write("Результат")
    if predictions[1] > 0.7:
        st.error("neutral {:.2f} toxic {:.2f}".format(predictions[0], predictions[1]))
    elif predictions[1] <= 0.7 and predictions[1] >= 0.4:
        st.warning("neutral {:.2f} toxic {:.2f}".format(predictions[0], predictions[1]))
    elif predictions[1] < 0.4 and predictions[1] >= 0:
        st.success("neutral {:.2f} toxic {:.2f}".format(predictions[0], predictions[1]))

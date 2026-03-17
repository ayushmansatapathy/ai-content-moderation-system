from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

app = FastAPI()

labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("../model/tokenizer")

# Load model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=6
)

model.load_state_dict(torch.load("../model/toxicity_model.pth", map_location=torch.device("cpu")))

model.eval()


class Comment(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "AI Content Moderation API Running"}


@app.post("/predict")
def predict(comment: Comment):

    inputs = tokenizer(
        comment.text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    probs = torch.sigmoid(outputs.logits).numpy()[0]

    result = {labels[i]: float(probs[i]) for i in range(len(labels))}

    return result
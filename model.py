from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from pathlib import Path


BASE_DIR = Path(__file__).resolve(strict=True).parent

# load the saved  pretrained fine-tuned model
model_name = fr"{BASE_DIR}/PipelineModel"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def predict(text):
    batch = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        output = model(**batch)
        probability = F.softmax(output.logits, dim=1)
        labels = torch.argmax(probability, dim=1)
        label = [model.config.id2label[label] for label in labels.tolist()]
        return label   #, probability.tolist()


# text = "I like this product", "good"
# sentiment, prediction = predict(text)
# print(sentiment)
# print(prediction)
# print(type(sentiment))
# print(type(prediction))


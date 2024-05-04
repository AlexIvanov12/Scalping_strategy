from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch

# Load the pre-trained model and tokenizer
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

def sentiment_analysis(text):
    input = tokenizer(text, return_tensors = "pt", padding = True, truncation = True,  max_length=512)
    with torch.no_grad():
        logits = model(**input).logits

    probabilities = softmax(logits, dim = 1).squeeze()
    sentiment = torch.argmax(probabilities).item()
    labels = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
    return labels[sentiment], probabilities.numpy()

text = "The company's profits have decreased, leading to a disappointing quarter."
sentiment, probabilities = sentiment_analysis(text)
print(f'Sentiment: {sentiment}, Probabilities: {probabilities}')
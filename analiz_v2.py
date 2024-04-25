from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, BertForSequenceClassification, BertTokenizer
import torch
import scipy
from sklearn.metrics import accuracy_score

# Завантаження необхідних ресурсів NLTK
nltk.download('vader_lexicon')

# Ініціалізація SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Створення DataFrame з даними
data = {'Text': ['I love cookie', 'I hate rain', 'I am indifferent about the weather']}
df = pd.DataFrame(data)

# Аналіз настроїв для кожного тексту у DataFrame
predicted_sentiments = []
for text in df['Text']:
    score = sia.polarity_scores(text)
    if score['compound'] >= 0.05:
        predicted_sentiments.append('positive')
    elif score['compound'] <= -0.05:
        predicted_sentiments.append('negative')
    else:
        predicted_sentiments.append('neutral')

df['predicted_sia'] = predicted_sentiments

# Підготовка і використання моделі FinBERT
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model_finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
X = df['Text']  # Припускаючи, що X має бути текстами з df

preds = []
preds_proba = []
tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 512}
for x in X:
    with torch.no_grad():
        input_sequence = tokenizer(x, return_tensors="pt", **tokenizer_kwargs)
        logits = model_finbert(**input_sequence).logits
        scores = {k: v for k, v in zip(model_finbert.config.id2label.values(), scipy.special.softmax(logits.numpy().squeeze()))}
        sentimentFinbert = max(scores, key=scores.get)
        probabilityFinbert = max(scores.values())
        preds.append(sentimentFinbert)
        preds_proba.append(probabilityFinbert)

# Інші моделі аналізу настроїв та оцінка точності можуть бути включені аналогічно
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

results = df['Text'].apply(lambda x: nlp(x)[0])

def evaluate_model_accuracy(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Model: {model_name} - Accuracy-Score: {accuracy:.4f}')

# Припустимо, що у нас є фактичні мітки, які ми можемо порівняти з передбаченнями
y_true = ['positive', 'negative', 'neutral']  # це приклад, потрібно використовувати справжні мітки

evaluate_model_accuracy('SentimentIntensityAnalyzer', y_true, predicted_sentiments)
evaluate_model_accuracy('ProsusAI/finbert', y_true, preds)
#evaluate_model_accuracy('cardiffnlp/twitter-roberta-base-sentiment', y_true, preds_roberta)  # Розкоментуйте, коли змінна preds_roberta буде визначена
#evaluate_model_accuracy('yiyanghkust/finbert-tone', y_true, df.predicted_finbertTone)  # Розкоментуйте, коли df.predicted_finbertTone буде визначено

# Вивід результатів
print("Predicted sentiments:", predicted_sentiments)






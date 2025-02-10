import json
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def load_intents(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def train_and_save_model(intents_file, model_path="model.pkl"):
    print("Training new model...")

    data = load_intents(intents_file)

    X = []  # list of input texts
    y = []  # list of labels (intent tags)
    for intent in data["intents"]:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            X.append(pattern)
            y.append(tag)

    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("nb", MultinomialNB())])

    pipeline.fit(X, y)

    model_data = {"pipeline": pipeline, "intents_data": data}
    joblib.dump(model_data, model_path)
    print(f"Model trained and saved to {model_path}")

    return pipeline, data


def predict_class(pipeline, text):
    return pipeline.predict([text])[0]


if __name__ == "__main__":
    train_and_save_model("intents.json")
    

import json
import random
import joblib
from train_chatbot import MultinomialNaiveBayes

with open("model.pkl", "rb") as f:
    data_dict = joblib.load(f)

nb_model = data_dict["model"]
vocab = data_dict["vocab"]
idf = data_dict["idf_vals"]

with open("intents.json", "r") as file:
    dataset = json.load(file)

while True:
    user_input = input("\nEnter text (or type 'exit' to quit): ").strip()
    if user_input.lower() == "exit":
        break

    predicted_intent = nb_model.predict_class(vocab, idf, user_input)
    print(f"Predicted intent: {predicted_intent}")

    for intent in dataset["intents"]:
        if intent["tag"] == predicted_intent:
            print(f"Response: {random.choice(intent['responses'])}")
            print("-----------------")
            print()

            break

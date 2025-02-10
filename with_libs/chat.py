import json
import random
import joblib
import train_chatbot

with open("model.pkl", "rb") as f:
    data_dict = joblib.load(f)

model_pipeline = data_dict["pipeline"]

with open("intents.json", "r") as file:
    dataset = json.load(file)

while True:
    user_input = input("\nEnter text (or type 'exit' to quit): ").strip()
    if user_input.lower() == "exit":
        break

    # 3. Predict intent
    predicted_intent = train_chatbot.predict_class(model_pipeline, user_input)
    print(f"Predicted intent: {predicted_intent}")

    # 4. Provide response from dataset
    for intent in dataset["intents"]:
        if intent["tag"] == predicted_intent:
            print(f"Response: {intent['responses'][0]}")
            print("-----------------")
            break

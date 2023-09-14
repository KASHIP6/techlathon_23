import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

from flask import Flask, request, jsonify, render_template  # Import render_template

app = Flask(__name__)
app.config['DEBUG'] = True  # Enable debugging mode

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intent data from Categories.json
intents = json.loads(open('Categories.json').read())

# Load preprocessed data, classes, and trained chatbot model
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot.h5')

# Function to clean up a sentence using tokenization and lemmatization
def cleanUpSentence(sentence):
    sentenceWords = nltk.word_tokenize(sentence)
    sentenceWords = [lemmatizer.lemmatize(word) for word in sentenceWords]
    return sentenceWords

# Function to convert a sentence into a bag of words
def bagOfWords(sentence):
    sentenceWords = cleanUpSentence(sentence)
    bag = [0] * len(words)
    for w in sentenceWords:
        for i, word in enumerate(words):
            if w == word:
                bag[i] = 1
    return np.array(bag)

# Function to predict the class (intent) of a sentence
def predictClass(sentence):
    bow = bagOfWords(sentence)
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    returnList = []
    for r in results:
        returnList.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return returnList

# Function to get a random response based on the predicted intent
def getResponse(intentsList, intentsJson):
    tag = intentsList[0]['intent']
    listOfIntents = intentsJson['categories']
    for i in listOfIntents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Define a route to render the index.html file
@app.route('/')
def index():
    return render_template('index.html')


# Define an API route for receiving user messages and sending responses
@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.json['message']

    # Use your existing code for processing user input and generating responses
    ints = predictClass(user_message)
    bot_response = getResponse(ints, intents)

    return jsonify({'response': bot_response})

# Define an API route for training the chatbot (triggered manually or via an endpoint)
@app.route('/train', methods=['POST'])
def train_chatbot():
    # Your training script (training.py) can be executed here
    # Ensure the 'chatbot.h5' model file is saved upon completion
    return jsonify({'message': 'Training initiated. Check logs for progress.'})

if __name__ == '__main__':
    app.run(debug=True)

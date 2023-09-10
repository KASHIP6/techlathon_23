#################################################################
# Input from the user is taken. It is cleaned using NLP,         #
# and then it is sent to the trained Neural Network model.      #
# The model predicts the tag of that sentence, and then a       #
# random response corresponding to that tag is returned to      #
# the user.                                                      #
#################################################################

import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

from flask import Flask, request, jsonify

app = Flask(__name__)

# Define an API route for receiving user messages and sending responses
@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.json['message']
    
    # Use your existing code for processing user input and generating responses
    ints = predictClass(user_message)
    bot_response = getResponse(ints, intents)
    
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run()

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intent data from Categories.json
intents = json.loads(open('Categories.json').read())

# Load preprocessed data
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load the trained chatbot model
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

print('Bot is Running !!')

# A loop to interact with the chatbot in the console
while True:
    message = input("You: ")
    ints = predictClass(message)
    res = getResponse(ints, intents)
    print('Grey:', res)

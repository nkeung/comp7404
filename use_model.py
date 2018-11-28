
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
import json
import keras
import keras.preprocessing.text as processText
import numpy as np

# need to load the word token
dataToken = Tokenizer(num_words=3000)
categories = ['negative', 'positive']

# load the words from file
with open('storeWord.txt', 'r') as textFile:
    dataDic = json.load(textFile)

def make_array(text):
    words = processText.text_to_word_sequence(text)
    wordIdx = []
    for word in words:
        wordIdx.append(dataDic[word])
    return wordIdx

# load the model
model_file = open('layers.txt', 'r')
model = model_file.read()
model_file.close()
model = model_from_json(model)
model.load_weights('layers.weights')

# ask for user input
user_input = input('Input a sentence: ')
prediction = model.predict(dataToken.sequences_to_matrix([make_array(user_input)], mode='binary'))
# and print it for the humons
print("%s sentiment is %f%% percent!" % (categories[np.argmax(prediction)], prediction[0][np.argmax(prediction)] * 100))
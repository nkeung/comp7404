import json
import keras
import keras.preprocessing.text as kpt
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

# parse the data, skip the header and only choose second and fouth columns
data = np.genfromtxt('data.csv', delimiter=',', skip_header=1, usecols=(1, 3), dtype=None, encoding='utf8')

# store data from column 3
dataX = [x[1] for x in data]

# store labels from column 1
dataY = np.array([x[0] for x in data])

# initialize dataToken to have 3000 spaces
dataToken = Tokenizer(num_words=3000)
dataToken.fit_on_texts(dataX)

# Store words in a file so we can get it later
with open('storeWord.txt', 'w') as textFile:
    json.dump(dataToken.word_index, textFile)

def make_array(text):
    # make word to array
    return [dataToken.word_index[word] for word in kpt.text_to_word_sequence(text)]

# store word index
wordIdx = []
for text in dataX:
    wordIdx.append(make_array(text))

wordIdx = np.array(wordIdx)

# if the word is included in the 3000 words then assign 1, if not, then assign 0
dataX = dataToken.sequences_to_matrix(wordIdx, mode='binary')
# positive and negative category
dataY = keras.utils.to_categorical(dataY, 2)

# create nerual network layers
layers = Sequential()
layers.add(Dense(512, input_shape=(3000,), activation='relu'))
layers.add(Dropout(0.5))
layers.add(Dense(256, activation='relu'))
layers.add(Dropout(0.5))
layers.add(Dense(2, activation='softmax'))

layers.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

layers.fit(dataX, dataY, batch_size=32, epochs=5, verbose=1, validation_split=0.1, shuffle=True)

# save the model to a file
model_txt = layers.to_json()
with open('layers.txt', 'w') as save_result:
    save_result.write(model_txt)

layers.save_weights('layers.weights')

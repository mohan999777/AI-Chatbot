import json
import numpy as np
import tensorflow

import random
import nltk
from nltk.stem import WordNetLemmatizer

lematize = WordNetLemmatizer()


intents = json.loads(open("/home/mohan/input.json").read())

words =[]
classes =[]

documents=[]

ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:

    for promp in intent['Prompt']:

        word_list = nltk.word_tokenize(promp)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])
print(documents)

words = [lematize.lemmatize(word)  for word in words  if word not in ignore_letters]

words = sorted(set(words))

classes = sorted(set(classes))

print(words)

import pickle

pickle.dump(words, open('words.pk', 'wb'))
pickle.dump(classes, open('classes.pk', 'wb'))


training =[]

output_empty = [0]*len(classes)


for document in documents:
    bag=[]
    word_patterns = document[0]
    word_patterns = [lematize.lemmatize(word.lower())  for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)

    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)    

training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation, Dropout

from tensorflow.keras.optimizers import SGD

model = Sequential()

model.add(Dense(128, input_shape=(len(train_x[0]),), activation ='relu'))

model.add(Dropout(0.5))

model.add(Dense(64, activation ='relu'))

model.add(Dropout(0.5))

model.add(Dense(len(train_y[0]), activation ='sigmoid'))

# sgd = SGD(learning_rate=0.01, decay= 1e-6, momentum=0.9, nestrov= True)

# sgd = tensorflow.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')

adam= tensorflow.keras.optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')

model.compile(loss= 'categorical_crossentropy', optimizer= adam, metrics= ['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs= 100, batch_size= 4, verbose= 1 )

model.save('chatbot_model.h5', hist)

print('Done')




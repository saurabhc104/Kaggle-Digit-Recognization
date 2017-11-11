#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 17:24:14 2017

@author: saurabh
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot(a, label):
    a = np.reshape(a,(28,28))
    plt.imshow(a)
    plt.suptitle(label)

#seed = 7
#np.random.seed(seed)
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X_train = train[train.columns[1:]]
Y_train = train[train.columns[0]]
#index = 5
#plot(X_train.iloc[index], Y_train[index])
X_test = test[test.columns[:]]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


#One hot encoding
encoded = np.zeros((len(X_train),10))
for i in range(len(X_train)):
    encoded[i][Y_train[i]] = 1
    

model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
#model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer=RMSprop(), metrics=['accuracy'])

history = model.fit(X_train.as_matrix(), encoded, epochs=20, batch_size=512, verbose=1)
plt.plot(history.history['loss'])

result = model.predict(X_test.as_matrix())

#Reverse one hot encoding
decoded = np.zeros((len(X_test)))
for i in range(len(X_test)):
    for j in range(10):
        if result[i][j] == 1 :
            decoded[i] = j
            break
    



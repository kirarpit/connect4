#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 21:19:49 2018

@author: Arpit
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

ann = Sequential()
ann.add(Dense(units = 4, kernel_initializer = "uniform", activation = 'relu', input_dim = 2))
ann.add(Dense(units = 2, kernel_initializer = "uniform", activation = 'relu'))
ann.add(Dense(units = 2, kernel_initializer = "uniform", activation = 'softmax'))
ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

X_train = np.array([[1,1], [0,1], [1,0], [0,0]])
y_train = np.array([[0.4, 0], [0, 0.4], [0, 0.4], [0.4,0]])


cnt = 0
while True:
    cnt += 1
    
    verbosity = 0
    if cnt % 1 == 0:
        verbosity = 2
        print "Iteration: " + str(cnt)
        
    result = ann.fit(X_train, y_train, verbose=verbosity, epochs = 50, batch_size=4)
    
    if result.history["loss"][0] <= 0.008:
        break
    
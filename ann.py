#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:53:30 2018

@author: Arpit
"""

from keras.models import Sequential
from keras.layers import Dense

class ANN:
    def __init__(self):
        self.createANN()
    
    def createANN(self):
        ann = Sequential()
        ann.add(Dense(units = 42, kernel_initializer = "uniform", activation = 'relu', input_dim = 84))
        ann.add(Dense(units = 20, kernel_initializer = "uniform", activation = 'relu'))
        ann.add(Dense(units = 7, kernel_initializer = "uniform", activation = 'softmax'))
        ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        
        self.ann = ann
    
    def save(self, filename):
        self.ann.save_weights(filename + '_weights')
        
        json = self.ann.to_json()
        file = open(filename + "_config.json", "w")
        file.write(json)
        file.close()
        
    def load(self, filename):
        self.ann.load_weights(filename + '_weights', by_name=False)

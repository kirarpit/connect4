#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:53:30 2018

@author: Arpit
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import os.path

class Brain:
    def __init__(self, name, game, model=None):
        self.filename = str(name) + '.h5'
        self.stateCnt, self.actionCnt = game.getStateActionCnt()
        
        if model is None:
            self.model = self._buildModel()
        else:
            print("Custom ANN model loaded")
            self.model = model
    
    def _buildModel(self):
        model = Sequential()
        model.add(Dense(units = int((self.stateCnt + self.actionCnt)/2),
                      kernel_initializer='random_uniform',
                      bias_initializer='random_uniform',
                      activation = 'relu',
                      input_dim = self.stateCnt))
        model.add(Dense(units = self.actionCnt,
                      kernel_initializer='random_uniform',
                      bias_initializer='random_uniform',
                      activation = 'linear'))
        model.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])
        return model
    
    def setModel(self, model):
        self.model = model
        
    def predict(self, s):
        return self.model.predict(s)

    def save(self):
        self.model.save(self.filename)
        
    def load(self, filename):
        filename = str(filename) + '.h5'
        if os.path.exists(filename):
            print (filename + " model loaded")
            self.model = load_model(filename)
        else:
            print("Error: file " + filename + " not found")
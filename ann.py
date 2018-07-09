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

#from keras.regularizers import l2, l1
#from keras.layers.advanced_activations import LeakyReLU

class ANN:
    def __init__(self, name, game):
        self.filename = str(name) + '.h5'
        self.createANN(game)
        self.load()
    
    def createANN(self,game):
        ann = Sequential()
        ann.add(Dense(units = 42, kernel_initializer = "he_normal", activation = 'relu', input_dim = game.columns * game.rows * 2))
        ann.add(Dense(units = 42, kernel_initializer = "he_normal", activation = 'relu'))
        ann.add(Dense(units = game.columns, kernel_initializer = "he_normal", activation = 'linear'))
        ann.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])
        self.ann = ann
    
    def save(self):
        self.ann.save(self.filename)
        
    def load(self):
        if os.path.exists(self.filename):
            print (self.filename + " model loaded")
            self.ann = load_model(self.filename)
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
        self.filename = str(name) + '.model'
        self.createANN(game)
        self.load()
    
    def createANN(self,game):
        ann = Sequential()
#        ann.add(Dense(units = 42, kernel_regularizer = l2(0.01), kernel_initializer = "he_normal", activation = 'relu', input_dim = 84))
        ann.add(Dense(units = 42, kernel_initializer = "he_normal", activation = 'relu', input_dim = game.columns * game.rows * 2))
        ann.add(Dense(units = 42, kernel_initializer = "he_normal", activation = 'relu'))
#        ann.add(LeakyReLU(alpha=0.3))

        ann.add(Dense(units = 7, kernel_initializer = "he_normal", activation = 'linear'))
        ann.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['accuracy'])
#        ann.compile(optimizer = 'adam', loss = 'kullback_leibler_divergence', metrics = ['categorical_accuracy'])
#        ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['categorical_accuracy'])
        self.ann = ann
    
    def save(self):
        self.ann.save(self.filename)
        
    def load(self):
        print (self.filename)
        if os.path.exists(self.filename):
            print ("model loaded")
            self.ann = load_model(self.filename)
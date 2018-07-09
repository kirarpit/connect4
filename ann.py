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
import keras
import tensorflow as tf

#from keras.regularizers import l2, l1
#from keras.layers.advanced_activations import LeakyReLU
            
def huber_loss(y_true, y_pred, clip_delta=1.0):
  error = y_true - y_pred
  cond  = keras.backend.abs(error) < clip_delta

  squared_loss = 0.5 * keras.backend.square(error)
  linear_loss  = clip_delta * (keras.backend.abs(error) - 0.5 * clip_delta)

  loss = tf.where(cond, squared_loss, linear_loss)
  return keras.backend.mean(loss)
  
class ANN:
    def __init__(self, name, game):
        self.filename = str(name) + '_1.h5'
        self.createANN(game)
        self.load()
    
    def createANN(self,game):
        ann = Sequential()
        ann.add(Dense(units = 42, activation = 'relu', input_dim = game.columns * game.rows * 2))
        ann.add(Dense(units = 42, activation = 'relu'))
        ann.add(Dense(units = game.columns, activation = 'linear'))
        ann.compile(optimizer = 'rmsprop', loss = huber_loss, metrics = ['accuracy'])
        self.ann = ann
    
    def save(self):
        self.ann.save(self.filename)
        
    def load(self):
        if os.path.exists(self.filename):
            print (self.filename + " model loaded")
            self.ann = load_model(self.filename)

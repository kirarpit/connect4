#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:53:30 2018

@author: Arpit
"""

from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from keras.models import load_model
import os.path
import keras
import tensorflow as tf

def huber_loss(y_true, y_pred, clip_delta=1.0):
  error = y_true - y_pred
  cond  = keras.backend.abs(error) < clip_delta

  squared_loss = 0.5 * keras.backend.square(error)
  linear_loss  = clip_delta * (keras.backend.abs(error) - 0.5 * clip_delta)

  loss = tf.where(cond, squared_loss, linear_loss)
  return keras.backend.mean(loss)

class ANN:
    def __init__(self, name, stateCnt, actionCnt):
        self.filename = str(name) + '.h5'
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.createANN()
        self.load()

    def createANN(self):
        ann = Sequential()
#        ann.add(Convolution2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', input_shape=self.stateCnt, data_format="channels_first"))
#        ann.add(MaxPooling2D(pool_size = (2, 2), strides=2, padding='same', data_format="channels_first"))
#        ann.add(Flatten())
        ann.add(Dense(units = 48, 
                      kernel_initializer='random_uniform', 
                      bias_initializer='random_uniform', 
                      activation = 'relu',
                      input_dim = self.stateCnt))
        ann.add(Dense(units = 24, 
                      kernel_initializer='random_uniform', 
                      bias_initializer='random_uniform', 
                      activation = 'relu'))
        ann.add(Dense(units = self.actionCnt, 
                      kernel_initializer='random_uniform', 
                      bias_initializer='random_uniform', 
                      activation = 'linear'))
        ann.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])
        self.ann = ann
    
    def save(self):
        self.ann.save(self.filename)
        
    def load(self):
        if os.path.exists(self.filename):
            print (self.filename + " model loaded")
            self.ann = load_model(self.filename, custom_objects={'huber_loss': huber_loss})

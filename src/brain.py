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
import tensorflow as tf
from keras import backend as K

class Brain:
    def __init__(self, name, game, **kwargs):
        self.filename = str(name) + '.h5'
        self.stateCnt, self.actionCnt = game.getStateActionCnt()
        
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        if "model" in kwargs and kwargs['model'] is not None:
            self.model = kwargs['model']
        else:
            self.model = self._build_model()
        
        self.model._make_predict_function()
        self.model._make_train_function()
        
        self.session.run(tf.global_variables_initializer())
        if "loadWeights" in kwargs and kwargs['loadWeights']:
            self.load_weights()

        self.default_graph = tf.get_default_graph()

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
    
    def predict(self, s):
        with self.default_graph.as_default():
            return self.model.predict(s)

    def train(self, x, y, batch_size, verbose):
        with self.default_graph.as_default():
            self.model.fit(x, y, batch_size=batch_size, verbose=verbose)
        
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)
    
    def save(self):
        self.model.save(self.filename)
        
    def load_weights(self):
        if os.path.exists(self.filename):
            print (self.filename + " weights loaded")
            self.model.load_weights(self.filename)

    def load(self, filename):
        filename = str(filename) + '.h5'
        if os.path.exists(filename):
            print (filename + " model loaded")
            self.model = load_model(filename)
        else:
            print("Error: file " + filename + " not found")
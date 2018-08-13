#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:53:30 2018

@author: Arpit
"""

from brains.brain import Brain
from keras.models import Sequential
from keras.layers import Dense
import os.path
import tensorflow as tf
from keras import backend as K

class QBrain(Brain):
    def __init__(self, name, game, **kwargs):
        super().__init__(name, game, **kwargs)

        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model._make_predict_function()
        self.model._make_train_function()
        
        self.session.run(tf.global_variables_initializer())
        if "load_weights" in kwargs and kwargs['load_weights']: self.load_weights()

        self.default_graph = tf.get_default_graph()

    def _build_model(self):
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
        

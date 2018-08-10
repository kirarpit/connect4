#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 12:23:29 2018

@author: Arpit
"""

from brains.brain import Brain
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import numpy as np

LEARNING_RATE = 1e-3
BATCH_SIZE = 256
EPOCHS = 10

class ZeroBrain(Brain):
    def __init__(self, name, game, **kwargs):
        super().__init__(name, game, **kwargs)
        self.lr = kwargs['lr'] if "lr" in kwargs else LEARNING_RATE
        self.batch_size = kwargs['batch_size'] if "batch_size" in kwargs else BATCH_SIZE
        self.epochs = kwargs['epochs'] if "epochs" in kwargs else EPOCHS

    def _buildModel(self):
        l_input = Input( batch_shape=(None, self.stateCnt) )
        l_dense = Dense(24, kernel_initializer='random_uniform', bias_initializer='random_uniform', 
                        activation='relu')(l_input)
        l_dense = Dense(24, kernel_initializer='random_uniform', bias_initializer='random_uniform', 
                        activation='relu')(l_dense)
        
        out_actions = Dense(self.actionCnt, activation='softmax', name="P")(l_dense)
        out_value   = Dense(1, activation='linear', name="V")(l_dense)
        
        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model.compile(loss=['categorical_crossentropy','mean_squared_error'], 
                      optimizer=Adam(self.lr))

        return model
    
    def predict(self, s):
        P, V = self.model.predict(s)
        return P[0], V[0][0]

    def train(self, memory):
        states, Ps, Vs = list(zip(*memory))
        states = np.asarray(states)
        Ps = np.asarray(Ps)
        Vs = np.asarray(Vs)
        self.model.fit(x = states, y = [Ps, Vs], batch_size = self.batch_size, 
                       epochs = self.epochs, verbose=2)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 12:23:29 2018

@author: Arpit
"""

from brains.brain import Brain
from keras.models import Model
from keras.layers import Input, Flatten, LeakyReLU
from keras.optimizers import Adam
import numpy as np
import time, threading
import tensorflow as tf
from keras import backend as K

class ZeroBrain(Brain):
    def __init__(self, name, game, **kwargs):
        super().__init__(name, game, **kwargs)
        
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)
        
        self.model._make_predict_function()
        self.model._make_train_function()

        self.session.run(tf.global_variables_initializer())

        self.threading = kwargs['threading'] if "threading" in kwargs else False
        if "load_weights" in kwargs and kwargs['load_weights']: self.load_weights()
        self.memories = []
        self.lock = threading.Lock()

        self.default_graph = tf.get_default_graph()

    def _build_model(self):
        if self.conv:
            main_input, x = self.get_conv_layers()
            out_value = self.value_head(x)
            out_actions = self.policy_head(x)
            
        else:
            main_input = Input( batch_shape=(None, self.stateCnt) )
            
            x = main_input
            if len(self.layers) > 0:
                for h in self.layers:
                    x = self.dense_layer(x, h['size'])
                    x = LeakyReLU()(x)

            out_value = self.dense_layer(x, 1, 'tanh', 'value_head')
            out_actions = self.dense_layer(x, self.actionCnt, 'softmax', 'policy_head')

        model = Model(inputs=[main_input], outputs=[out_actions, out_value])
        model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': 'categorical_crossentropy'},
                      loss_weights={'value_head': 0.5, 'policy_head': 0.5},
                      optimizer=Adam(self.learning_rate))

        return model
    
    def value_head(self, x):
        x = self.conv_layer(x, 1, (1,1))
        x = Flatten()(x)
        x = self.dense_layer(x, self.actionCnt)
        x = LeakyReLU()(x)
        x = self.dense_layer(x, 1, 'tanh', 'value_head')
        return x
    
    def predict(self, s):
        with self.default_graph.as_default():
            P, V = self.model.predict(s)
            return P[0], V[0][0]

    def addToMem(self, memory):
        if self.threading:
            self.memories.append(memory)
        else:
            self.train(memory)
            
    def optimize(self):
        if len(self.memories) == 0:
            time.sleep(1)
            return

        if len(self.memories) > 100: print("Training alert! Increase threads!")
        
        memory = self.memories.pop()
        print("Training! Remaining length of memories %d", len(self.memories))
        self.train(memory)
        
    def train(self, memory):
        with self.lock:
            states, Ps, Vs = list(zip(*memory))
            states = np.asarray(states)
            Ps = np.asarray(Ps)
            Vs = np.asarray(Vs)
    
        with self.default_graph.as_default():
            self.model.fit(x = states, y = [Ps, Vs], batch_size = self.batch_size, 
                       epochs = self.epochs, verbose=2)
            
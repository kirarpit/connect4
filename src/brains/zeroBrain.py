#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 12:23:29 2018

@author: Arpit
"""

from brains.brain import Brain, softmax_cross_entropy_with_logits
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import SGD, Adam
import numpy as np

HIDDEN_CNN_LAYERS = [
	{'filters':25, 'kernel_size': (4,4)}
	 , {'filters':25, 'kernel_size': (4,4)}
	 , {'filters':25, 'kernel_size': (4,4)}
	]

class ZeroBrain(Brain):
    def __init__(self, name, game, **kwargs):
        super().__init__(name, game, **kwargs)
        
        if self.hidden_layers is None:
            self.hidden_layers = HIDDEN_CNN_LAYERS
            
    def _build_model(self):
        if self.conv:
            main_input = Input(shape = self.stateCnt, name = 'main_input')
            x = self.conv_layer(main_input, self.hidden_layers[0]['filters'], 
                                self.hidden_layers[0]['kernel_size'])
            
            if len(self.hidden_layers) > 1:
                for h in self.hidden_layers[1:]:
                    x = self.residual_layer(x, h['filters'], h['kernel_size'])
                    
            out_value = self.value_head(x)
            out_actions = self.policy_head(x)

        else:
            main_input = Input( batch_shape=(None, self.stateCnt) )
            l_dense = Dense(24, kernel_initializer='random_uniform', bias_initializer='random_uniform', 
                        activation='relu')(main_input)
            l_dense = Dense(24, kernel_initializer='random_uniform', bias_initializer='random_uniform', 
                        activation='relu')(l_dense)
            out_actions = Dense(self.actionCnt, activation='softmax', name="policy_head")(l_dense)
            out_value   = Dense(1, activation='tanh', name="value_head")(l_dense)

        model = Model(inputs=[main_input], outputs=[out_actions, out_value])
        model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': 'categorical_crossentropy'},
#        model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': softmax_cross_entropy_with_logits},
                      optimizer=Adam(self.learning_rate),
#                      optimizer=SGD(lr=self.learning_rate, momentum = self.momentum),
                      loss_weights={'value_head': 0.5, 'policy_head': 0.5})

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
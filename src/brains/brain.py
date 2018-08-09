#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 12:30:56 2018

@author: Arpit
"""
import os.path
from keras.models import load_model

class Brain:
    def __init__(self, name, game, **kwargs):
        self.filename = str(name) + '.h5'
        self.stateCnt, self.actionCnt = game.getStateActionCnt()
        
        self.batch_size = kwargs['batch_size'] if "batch_size" in kwargs else 64
        self.epochs = kwargs['epochs'] if "epochs" in kwargs else 1

        self.gamma = kwargs['gamma'] if "gamma" in kwargs else 0.99
        self.n_step = kwargs['n_step'] if "n_step" in kwargs else 1
        self.gamma_n = self.gamma ** self.n_step
        self.min_batch = kwargs['min_batch'] if "min_batch" in kwargs else 256
        
        if "model" in kwargs and kwargs['model'] is not None:
            self.model = kwargs['model']
        else:
            self.model = self._build_model()

    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)
    
    def load_weights(self):
        if os.path.exists(self.filename):
            print (self.filename + " weights loaded")
            self.model.load_weights(self.filename)

    def save(self):
        self.model.save(self.filename)
        
    def load(self, filename=None):
        if filename is None:
            filename = self.filename
        else:
            filename = str(filename) + '.h5'

        if os.path.exists(filename):
            print (filename + " model loaded")
            self.model = load_model(filename)
        else:
            print("Error: file " + filename + " not found")
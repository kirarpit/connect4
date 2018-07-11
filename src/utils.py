#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 18:26:43 2018

@author: Arpit
"""
import numpy as np

class Utils:

    @staticmethod
    def sample(probs):
        sum = 0.0
        r = np.random.uniform()
        for index, prob in enumerate(probs):
            sum += prob
            if sum > r:
                return index
    
        return index
    
    @staticmethod
    def normalize(x):
        x = x + abs(min(x))
        x = np.array([i/np.sum(x) for i in x])
        return x
    
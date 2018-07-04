#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 22:12:38 2018

@author: Arpit
"""
import random

class Memory:   # stored as ( s, a, r, s_ )

    def __init__(self, capacity):
        self.capacity = capacity
        self.samples = []
        self.rsamples = []

    def add(self, sample):
        samples = None
        if sample[2] != 0:
            samples = self.rsamples
        else:
            samples = self.samples
    
        samples.append(sample)        
        if len(samples) > self.capacity:
            samples.pop(0)

    def sample(self, n, ratio):
        n1 = ratio*n
        n2 = (1-ratio)*n
        
        n1 = int(min(n1, len(self.rsamples)))
        n2 = int(min(n2, len(self.samples)))

        return random.sample(self.rsamples, n1) + random.sample(self.samples, n2)
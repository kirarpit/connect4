#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 21:27:02 2018

@author: Arpit
"""
import matplotlib.pyplot as plt
import numpy as np

class QPlot:
    def __init__(self, ann, interval=1000):
        self.ann = ann
        self.interval = interval
        self.cnt = 0
        self.x = []
        self.states = np.array([
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,
        -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1,
        -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1,  1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1,  1, -1,
        -1, -1]
        ])
        self.actions = [3, 5]
        self.ys = np.empty((len(self.actions),), dtype=object)
        for i,v in enumerate(self.ys): self.ys[i] = list()
        
    def add(self):
        self.x.append(self.cnt * self.interval)
        preds = self.ann.predict(self.states)
        for index, pred in enumerate(preds):
            self.ys[index].append(pred[self.actions[index]])
        self.cnt += 1
        
    def show(self):
        for i in range(0, len(self.actions)):
            plt.plot(self.x, self.ys[i], label=i)
            
        plt.legend(loc = "best")
        plt.draw()
        plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 21:27:02 2018

@author: Arpit
"""
import matplotlib.pyplot as plt
import numpy as np
from game import Game

class QPlot:
    def __init__(self, game, ann, interval=1000):
        self.ann = ann
        self.interval = interval
        self.cnt = 0
        self.x = []
        self.states = np.empty([0, len(game.arrayForm[0]  )])
        self.actions = []
        self.getStatesAndActions(game)
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
        plt.savefig('/Users/Arpit/Desktop/qPlot.png')
        plt.draw()
        plt.show()

    def getStatesAndActions(self, game):
        g = Game(game.rows, game.columns)
        
        g.newGame()
        g.dropDisc(0)
        g.dropDisc(2)
        g.dropDisc(3)
        g.dropDisc(2)
        g.dropDisc(3)
        g.dropDisc(2)
        self.states = np.vstack([self.states, g.arrayForm])
        self.actions.append(2)
        
        self.states = np.vstack([self.states, g.arrayForm])
        self.actions.append(3)

        g.newGame()
        g.dropDisc(5)
        g.dropDisc(2)
        g.dropDisc(5)
        g.dropDisc(2)
        g.dropDisc(5)
        g.dropDisc(2)
        self.states = np.vstack([self.states, g.arrayForm])
        self.actions.append(5)

        self.states = np.vstack([self.states, g.arrayForm])
        self.actions.append(2)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 21:27:02 2018

@author: Arpit
"""
import matplotlib.pyplot as plt
import numpy as np
from games.c4Game import C4Game
from games.t3Game import T3Game

class QPlot:
    def __init__(self, name, stateCnt, actionCnt, ann, interval=1000):
        self.name = name
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.ann = ann
        self.interval = interval
        self.cnt = 0
        self.states = np.empty([0, stateCnt])
        self.getStatesC4()
#        self.getStatesT3()
        self.x = []
        self.ys = np.empty((self.states.shape[0],), dtype=object)
        for i,v in enumerate(self.ys): self.ys[i] = list()
        self.illMoves = None
        
    def add(self):
        self.x.append(self.cnt * self.interval)
        preds = self.ann.predict(self.states)
        for index, pred in enumerate(preds):
            if self.illMoves is not None:
                for illmove in self.illMoves[index]:
                    pred[illmove] = 0
                    
            self.ys[index].append(np.amax(pred))
        self.cnt += 1
        
    def show(self):
        for i in range(0, self.states.shape[0]):
            plt.plot(self.x, self.ys[i], label=i)
            
        plt.legend(loc = "best")
        plt.savefig('/Users/Arpit/Desktop/qPlot' + str(self.name) + '.png')
        plt.draw()
        plt.show()

    def printQValues(self):
        preds = self.ann.predict(self.states)
        for index, pred in enumerate(preds):
            print (str(np.amax(pred)) + ", ", end="")
        print ("\n")
        
    def getStatesT3(self):
        g = T3Game()
        self.illMoves = []
        
        if self.name == 1:
            #win
            g.newGame()
            g.step(4)
            g.step(5)
            self.states = np.vstack([self.states, g.getCurrentState()])
            self.illMoves.append([(4,5)])
            
            #lose
            g.newGame()
            g.step(3)
            g.step(0)
            g.step(6)
            g.step(1)
            self.states = np.vstack([self.states, g.getCurrentState()])
            self.illMoves.append([(3,0,6,1)])

            #Draw
            g.newGame()
            g.step(4)
            g.step(0)
            g.step(7)
            g.step(1)
            self.states = np.vstack([self.states, g.getCurrentState()])
            self.illMoves.append([(4,0,7,1)])
            
        else:
            #win
            g.newGame()
            g.step(3)
            g.step(0)
            g.step(6)
            self.states = np.vstack([self.states, g.getCurrentState()])
            self.illMoves.append([(3,0,6)])
            
            #lose
            g.newGame()
            g.step(4)
            g.step(5)
            g.step(0)
            self.states = np.vstack([self.states, g.getCurrentState()])
            self.illMoves.append([(4,5,0)])

            #draw
            g.newGame()
            g.step(4)
            self.states = np.vstack([self.states, g.getCurrentState()])
            self.illMoves.append([(4)])

    def getStatesC4(self):
        g = C4Game()
        
        if self.name == 1:
            #lose
            g.newGame()
            g.step(0)
            g.step(3)
            self.states = np.vstack([self.states, g.getCurrentState()])
    
            #draw
            g.newGame()
            g.step(2)
            g.step(3)
            self.states = np.vstack([self.states, g.getCurrentState()])
            
            #win
            g.newGame()
            g.step(3)
            g.step(3)
            self.states = np.vstack([self.states, g.getCurrentState()])

        else:
            g.newGame()
            g.step(0)
            self.states = np.vstack([self.states, g.getCurrentState()])
    
            g.newGame()
            g.step(2)
            self.states = np.vstack([self.states, g.getCurrentState()])
            
            g.newGame()
            g.step(3)
            self.states = np.vstack([self.states, g.getCurrentState()])
            
            
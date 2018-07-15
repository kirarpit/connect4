#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 21:27:02 2018

@author: Arpit
"""
import numpy as np
from graphPlot import GraphPlot
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
        
        if type(self.stateCnt) is tuple:
            self.states = np.empty((0, *self.stateCnt[0:len(self.stateCnt)]))
        else:
            self.states = np.empty([0, stateCnt])
            
        self.getStatesC4()
#        self.getStatesT3()
        
        self.gPlot = GraphPlot("qPlot" + str(self.name), 1, self.states.shape[0], ["winState", "loseState", "drawState"])
        self.illMoves = None
        
    def add(self):
        x = self.cnt * self.interval
        y = []
        
        preds = self.ann.predict(self.states)
        for index, pred in enumerate(preds):
            if self.illMoves is not None:
                for illmove in self.illMoves[index]:
                    pred[illmove] = float("-inf")
                    
            y.append(np.amax(pred))
            
        self.gPlot.add(x, y)
        self.cnt += 1
        
    def show(self):
        self.gPlot.show()

    def getStatesT3(self):
        g = T3Game()
        self.illMoves = []
        
        if self.name == 1:
            #win
            g.newGame()
            g.step(4)
            g.step(5)
            self.states = np.vstack([self.states, g.getCurrentState()[np.newaxis]])
            self.illMoves.append([(4,5)])
            
            #lose
            g.newGame()
            g.step(3)
            g.step(0)
            g.step(6)
            g.step(1)
            self.states = np.vstack([self.states, g.getCurrentState()[np.newaxis]])
            self.illMoves.append([(3,0,6,1)])

            #Draw
            g.newGame()
            g.step(4)
            g.step(0)
            g.step(7)
            g.step(1)
            self.states = np.vstack([self.states, g.getCurrentState()[np.newaxis]])
            self.illMoves.append([(4,0,7,1)])
            
        else:
            #win
            g.newGame()
            g.step(3)
            g.step(0)
            g.step(6)
            self.states = np.vstack([self.states, g.getCurrentState()[np.newaxis]])
            self.illMoves.append([(3,0,6)])
            
            #lose
            g.newGame()
            g.step(4)
            g.step(5)
            g.step(0)
            self.states = np.vstack([self.states, g.getCurrentState()[np.newaxis]])
            self.illMoves.append([(4,5,0)])

            #draw
            g.newGame()
            g.step(4)
            self.states = np.vstack([self.states, g.getCurrentState()[np.newaxis]])
            self.illMoves.append([(4)])

    def getStatesC4(self):
        g = C4Game()
        
        if self.name == 1:
            #win
            g.newGame()
            g.step(3)
            g.step(3)
            self.states = np.vstack([self.states, g.getCurrentState()[np.newaxis]])

            #lose
            g.newGame()
            g.step(0)
            g.step(3)
            self.states = np.vstack([self.states, g.getCurrentState()[np.newaxis]])
    
            #draw
            g.newGame()
            g.step(2)
            g.step(3)
            self.states = np.vstack([self.states, g.getCurrentState()[np.newaxis]])
        else:
            #win
            g.newGame()
            g.step(0)
            self.states = np.vstack([self.states, g.getCurrentState()[np.newaxis]])
            
            #lose
            g.newGame()
            g.step(3)
            self.states = np.vstack([self.states, g.getCurrentState()[np.newaxis]])
    
            #draw
            g.newGame()
            g.step(2)
            self.states = np.vstack([self.states, g.getCurrentState()[np.newaxis]])
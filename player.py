#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:52:18 2018

@author: Arpit
"""
import ann
import os.path
import numpy as np

class Player:
    
    def __init__(self, name, game, debug):
        self.debug = debug
        if self.debug == True:
            self.epsilon = 0 #randomness
        else:
            self.epsilon = 0.5
        self.alpha = 0.5
        self.gamma = 0.92
        self.name = name
        self.ANN = ann.ANN()
        if os.path.exists(str(name)):
            self.ANN.load(str(name))
    
        self.gamerows = game.rows
        self.gamecolumns = game.columns
        self.resetLogs(0)
        
    def play(self, game, action=-1):
        if action != -1:
            game.dropDisc(action)
            game.printGameState()
            return action
        
        actions = self.ANN.ann.predict(game.arrayForm)
        if game.turnCnt > 2:
            prevQVal = self.y[-1:][0][self.moves[-1:]]
#            print "(" + str(np.max(actions[0])) + "*" + str(self.gamma) + "-" + str(prevQVal) + ")*" + str(self.alpha)
            prevQVal += (np.max(actions[0])*self.gamma - prevQVal)*self.alpha
            self.y[-1:][0][self.moves[-1:]] = prevQVal
        
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.gamecolumns, 1)
            action = action[0]
        else:
            action = np.argmax(actions[0])
        
        self.X_train = np.vstack([self.X_train, game.arrayForm])
        if self.debug == True:
            self.yo = np.vstack([self.yo, actions])
        self.y = np.vstack([self.y, actions])
        self.moves.append(action)
    
        game.dropDisc(action)

        return action
        
    def train(self):
        verb = 0
        if self.debug == True:
            verb = 2

        self.ANN.ann.fit(self.X_train, self.y, verbose=0, batch_size=50, epochs=100)
        self.resetLogs()
        
    def qUpdate(self, game):
        if self.name in game.rewards:
            reward = game.rewards[self.name]
            prevQVal = self.y[-1:][0][self.moves[-1:]]
#            print "(" + str(reward) + "-" + str(prevQVal) + ")*" + str(self.alpha)
            prevQVal += (reward - prevQVal)*self.alpha
            self.y[-1:][0][self.moves[-1:]] = prevQVal
        else:
            self.X_train = self.X_train[:-1]
            self.y = self.y[:-1]
        
    def resetLogs(self, oldLog=1):
        if self.debug == True:
            if oldLog != 0:
                self.x_old = np.copy(self.X_train)
                self.y_old = np.copy(self.y)
                self.yo_old = np.copy(self.yo)
                self.m_old = np.copy(self.moves)
        self.X_train = np.empty([0, self.gamerows * self.gamecolumns * 2], int)
        self.y = np.empty([0, self.gamecolumns], float)
        self.moves = []
        
        if self.debug == True:
            self.yo = np.copy(self.y)

    def saveExp(self):
        self.ANN.save(str(self.name))
        

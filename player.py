#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:52:18 2018

@author: Arpit
"""
import ann
import os.path
import numpy as np
import utils
from sys import exit

class Player:
    
    def __init__(self, name, game):
        self.name = name
        self.ANN = ann.ANN()
        if os.path.exists(str(name)):
            self.ANN.load(str(name))
    
        self.gamerows = game.rows
        self.gamecolumns = game.columns
        self.resetLogs()
        
    def play(self, game, move=-1):
        if move != -1:
            game.dropDisc(move)
            game.printGameState()
            return move
        
        cnt = 0
        while True:
            x = np.copy(game.arrayForm)
            probs = self.ANN.ann.predict(x)
            move = utils.Utils.sample(probs[0])
            result = game.dropDisc(move)
            
            #incase it's an illegal move
            if result == -2:
                if cnt > 100:
                    print probs
                
                probs[0][move] = 0
                probs[0] = utils.Utils.normalize(probs[0])
                self.ANN.ann.fit(x, probs, verbose=0, epochs=5, batch_size=1)
                cnt += 1
                
                if cnt > 100:
                    print "stuck in illegal move loop"
                    print probs
                    exit()
                continue
            
            self.X_train = np.vstack([self.X_train, x])
            self.y = np.vstack([self.y, probs])
            self.moves.append(move)
            break
        
        return move
        
    def train(self, game, verbosity):
        if game.winner == self.name:
            reward = 1
        else:
            reward = 0
        
        if game.winner != -3:
            for index, row in enumerate(self.y):
                if reward == 1:
                    row[True] = 0
                row[self.moves[index]] = reward
                
                if reward == 0:
                    row = utils.Utils.normalize(row)
                
            self.ANN.ann.fit(self.X_train, self.y, verbose=verbosity, batch_size=50)
            
        self.resetLogs()
        
    def resetLogs(self):
        self.X_train = np.empty([0, self.gamerows * self.gamecolumns * 2], int)
        self.y = np.empty([0, self.gamecolumns], float)
        self.moves = []

    def saveExp(self):
        self.ANN.save(str(self.name))
        

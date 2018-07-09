#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:52:18 2018

@author: Arpit
"""
import math
from ann import ANN
import numpy as np
from pMemory import PMemory
from qPlot import QPlot

BATCH_SIZE = 1000
GAMMA = 0.99
PLOT_INTERVAL = 10

#Exploration Rate
MIN_EPSILON = 0.01
MAX_EPSILON = 1
E_LAMBDA = 0.001

MEMORY_CAPACITY = 100000

class Player:
    
    def __init__(self, name, game, debug):
        self.debug = debug
        self.epsilon = 0 if self.debug else MAX_EPSILON
        self.name = name
        self.memory = PMemory(MEMORY_CAPACITY)
        self.stateCnt = game.rows * game.columns * 2
        self.nullState = np.zeros(self.stateCnt)
        self.ANN = ANN(name, game)
        self.tANN = ANN(str(name) + "_", game)
        self.updateTargetANN()
        self.qPlot = QPlot(self.ANN.ann, PLOT_INTERVAL)
        
    def play(self, game, action=-1):
        if action != -1:
            game.dropDisc(action)
            game.printGameState()
            return action
        
        s_ = np.copy(game.arrayForm[0]) if not game.isOver else None
        
        #observe
        if game.turnCnt > 1:
            r = game.rewards[self.name] if self.name in game.rewards else 0
            
            #add sample
            sample = (self.lastState, self.action, r, s_)
            x, y, errors = self.getTargets(game, [(0, sample)])
            self.memory.add(errors[0], sample)
            
            #train
            self.train(game)
        else:
            self.logs = {}

        #act
        if not game.isOver:
            if np.random.uniform() < self.epsilon:
                while True:
                    c = np.random.choice(game.columns, 1)[0]
                    if not game.isIllMove(c):
                        break
                self.action = c
            else:
                actions = self.ANN.ann.predict(game.arrayForm)[0]
                actions = self.filterIllMoves(game, actions)
                self.action = np.argmax(actions)
                
                if self.debug:
                    if 'preds' in self.logs:
                        self.logs['preds'] = np.vstack([self.logs['preds'], actions])
                        self.logs['moves'].append(self.action)
                    else:
                        self.logs['preds'] = actions
                        self.logs['moves'] = [self.action]
                    
            self.lastState = s_
            game.dropDisc(self.action)
            return self.action
        else:
            if game.gameCnt % PLOT_INTERVAL == 0:
                self.qPlot.add()
                self.qPlot.show()
                
            if game.gameCnt % 50 == 0:
                self.updateTargetANN()
                
            if not self.debug:
                self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-E_LAMBDA * game.gameCnt)
                
    def getTargets(self, game, batch):
        batchLen = len(batch)
        
        states = np.array([ o[1][0] for o in batch ])
        states_ = np.array([ (self.nullState if o[1][3] is None else o[1][3]) for o in batch ])

        p = self.ANN.ann.predict(states)
        p_ = self.ANN.ann.predict(states_)
        tp_ = self.tANN.ann.predict(states_)
        
        x = np.zeros((batchLen, self.stateCnt))
        y = np.zeros((batchLen, game.columns))
        errors = np.zeros(batchLen)

        for i in range(batchLen):
            o = batch[i][1]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * tp_[i][np.argmax(p_[i])] #DDQN

            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])
        
        return (x, y, errors)
        
    def train(self, game):
        batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self.getTargets(game, batch)
        
        #update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])
        
        verbosity = 1 if game.gameCnt % 100 == 0 else 0
        self.ANN.ann.fit(x, y, batch_size=100, verbose=verbosity)
            
        if self.debug and game.isOver:
            self.logs['x'] = x
            self.logs['y'] = y
            
    def updateTargetANN(self):
        self.tANN.ann.set_weights(self.ANN.ann.get_weights())

    def filterIllMoves(self, game, moves):
        for index, move in enumerate(moves):
            if game.isIllMove(index):
                moves[index] = float("-inf")
        
        return moves
    
    def saveWeights(self):
        self.ANN.save()
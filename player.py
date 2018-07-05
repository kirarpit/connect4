#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:52:18 2018

@author: Arpit
"""
import math
from ann import ANN
import numpy as np
from memory import Memory

BATCH_SIZE = 64
GAMMA = 0.99

#Exploration Rate
MIN_EPSILON = 0.01
MAX_EPSILON = 0.1
E_LAMBDA = 0.0005

class Player:
    
    def __init__(self, name, game, debug):
        self.debug = debug
        self.epsilon = 0 if self.debug else MAX_EPSILON
        self.name = name
        self.memory = Memory(10000)
        self.stateCnt = game.rows * game.columns * 2
        self.nullState = np.zeros(self.stateCnt)
        self.logs = {}
        self.ANN = ANN(name, game)

    def play(self, game, action=-1):
        if action != -1:
            game.dropDisc(action)
            game.printGameState()
            return action
        
        s_ = np.copy(game.arrayForm[0]) if not game.isOver else None
        if game.turnCnt >= 2:
            r = game.rewards[self.name] if self.name in game.rewards else 0
            if not game.isOver or r != 0:
                self.memory.add((self.lastState, self.action, r, s_))
            
            self.train(game)
        else:
            self.logs = {}

        if not game.isOver:
            if np.random.uniform() < self.epsilon:
                action = np.random.choice(game.columns, 1)
                self.action = action[0]
            else:
                actions = self.ANN.ann.predict(game.arrayForm)[0]
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
            if not self.debug:
                self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-E_LAMBDA * game.gameCnt)

    def train(self, game):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)
        
        states = np.array([ o[0] for o in batch ])
        states_ = np.array([ (self.nullState if o[3] is None else o[3]) for o in batch ])

        p = self.ANN.ann.predict(states)
        p_ = self.ANN.ann.predict(states_)
        
        if self.debug and game.isOver:
            self.logs['y_hat'] = np.copy(p)
        
        x = np.zeros((batchLen, self.stateCnt))
        y = np.zeros((batchLen, game.columns))

        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * np.amax(p_[i])

            x[i] = s
            y[i] = t
            
        if not self.debug:
            self.ANN.ann.fit(x, y, batch_size=64, verbose=0)
            
        if self.debug and game.isOver:
            self.logs['x'] = x
            self.logs['y'] = y
        
    def saveWeights(self):
        self.ANN.save()
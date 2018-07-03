#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:52:18 2018

@author: Arpit
"""
import math
from ann import ANN
import os.path
import numpy as np
from memory import Memory

BATCH_SIZE = 64
GAMMA = 0.99
MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001# speed of decay

class Player:
    
    def __init__(self, name, game, debug):
        self.debug = debug
        self.epsilon = 0 if self.debug == True else MAX_EPSILON
        self.alpha = 0.5
        self.name = name
        self.memory = Memory(100000)
        self.stateCnt = game.rows * game.columns * 2
        self.resetLogs(game, 0)

        self.ANN = ANN()
        if os.path.exists(str(name)):
            self.ANN.load(str(name))

    def play(self, game, action=-1):
        if action != -1:
            game.dropDisc(action)
            game.printGameState()
            return action
        
        if game.turnCnt >= 2:#first action now taken
            #observe
            s = self.X[-1:][0]
            a = self.moves[-1:]
            r = game.rewards[self.name] if self.name in game.rewards else 0
            s_ = np.copy(game.arrayForm[0]) if not game.isOver else None
           
            if not game.isOver or r != 0:
                self.memory.add((s, a, r, s_))
            
            #train with replay
            self.train(game)
                
        if not game.isOver:
            #select action for next iteration
            actions = self.ANN.ann.predict(game.arrayForm)[0]
            if np.random.uniform() < self.epsilon:
                action = np.random.choice(game.columns, 1)
                action = action[0]
            else:
                action = np.argmax(actions)
            
            #logging
            self.X = np.vstack([self.X, game.arrayForm])
            self.y = np.vstack([self.y, actions])
            self.moves.append(action)
        
            #take action
            game.dropDisc(action)
            return action
        else:
            if not self.debug:
                self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * game.gameCnt)
            self.resetLogs(game)
            
    def train(self, game):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)
        
        states = np.array([ o[0] for o in batch ])
        states_ = np.array([ (np.zeros(self.stateCnt) if o[3] is None else o[3]) for o in batch ])

        p = self.ANN.ann.predict(states)
        p_ = self.ANN.ann.predict(states_)
        
        if game.isOver and self.debug:
            self.p = np.copy(p)

        x = np.empty([0, self.stateCnt])
        y = np.empty([0, game.columns])

        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * np.amax(p_[i])

            x = np.vstack([x,s])
            y = np.vstack([y,t])
            
            self.ANN.ann.fit(x, y, verbose=0)
        
        if game.isOver and self.debug:
            self.batch = batch
            self.s = states
            self.s_ =  states_
            self.p_ = p_
            self.fx = x
            self.fy = y

    def resetLogs(self, game, oldLog=1):
        if self.debug == True:
            if oldLog != 0:
                self.x_old = np.copy(self.X)
                self.y_old = np.copy(self.y)
                self.m_old = np.copy(self.moves)
        self.X = np.empty([0, self.stateCnt], int)
        self.y = np.empty([0, game.columns], float)
        self.moves = []
        
    def saveWeights(self):
        self.ANN.save(str(self.name))
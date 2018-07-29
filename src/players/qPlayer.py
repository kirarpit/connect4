#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:52:18 2018

@author: Arpit
"""
from brain import Brain
import numpy as np
from memory.pMemory import PMemory
from mathEq import MathEq
from players.player import Player

#Discount Factor
GAMMA = 0.99

#Exploration Rate
MIN_EPSILON = 0.05
MAX_EPSILON = 1
E_LAMBDA = 0.001

#Learning Rate
MIN_ALPHA = 0.1
MAX_ALPHA = 0.5
A_LAMBDA = 0.001

MEMORY_CAPACITY = 20000
UPDATE_TARGET_FREQUENCY = 4000
BATCH_SIZE = 64
T_BATCH_SIZE = 64
PLOT_INTERVAL = UPDATE_TARGET_FREQUENCY/5

class QPlayer(Player):
    
    def __init__(self, name, game, **kwargs):
        super().__init__(name, game, **kwargs)
                
        self.nullState = np.zeros(self.stateCnt)
        self.epsilon = 0 if self.debug else MAX_EPSILON
        self.alpha = MAX_ALPHA
        self.verbosity = 0

        self.memory = PMemory(MEMORY_CAPACITY)
        self.goodMemory = PMemory(MEMORY_CAPACITY)
        
        model = kwargs['model'] if "model" in kwargs else None
        self.brain = Brain(name, game, model)
        self.tBrain = Brain(str(name) + "_", game, model)
        
        if self.eEq is None:
            self.eEq = MathEq({"min":MIN_EPSILON, "max":MAX_EPSILON, "lambda":E_LAMBDA})

        if self.aEq is None:
            self.aEq = MathEq({"min":MIN_ALPHA, "max":MAX_ALPHA, "lambda":A_LAMBDA})

        self.initLog()
        self.updateTargetBrain()
    
    def act(self, game):
        state = game.getCurrentState()
        illActions = game.getIllMoves()
        
        if np.random.uniform() < self.epsilon:
            action = self.getRandomMove(illActions)
        else:
            actions = self.brain.predict(np.array([state]))[0]
            fActions = self.filterIllMoves(np.copy(actions), illActions)
            action = np.argmax(fActions)
            
            if self.debug:
                self.logs['preds' + str(self.name)] = np.vstack([self.logs['preds' + str(self.name)], actions])
                self.logs['moves' + str(self.name)].append(action)
                
        return action

    def observe(self, sample, game):
        super().observe(game)
        gameCnt = game.gameCnt
        
        _, _, errors = self.getTargets([(0, sample)])
        memory = self.goodMemory if sample[2] > 0 else self.memory
        memory.add(errors[0], sample)
        
        if game.isOver():
            if gameCnt % UPDATE_TARGET_FREQUENCY == 0:
                self.updateTargetBrain()
                
        self.verbosity = 2 if gameCnt % PLOT_INTERVAL == 0 and sample[3] is not None else 0
        
    def train(self):
        batch = self.goodMemory.sample(int(BATCH_SIZE/2))
        goodMemLen = len(batch)
        
        batch += self.memory.sample(int(BATCH_SIZE - goodMemLen))
        x, y, errors = self.getTargets(batch)
        
        #update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            memory = self.goodMemory if i < goodMemLen else self.memory 
            memory.update(idx, errors[i])
        
        self.brain.train(x, y, T_BATCH_SIZE, self.verbosity)

        if self.debug:
            self.logs['x' + str(self.name)] = x
            self.logs['y' + str(self.name)] = y
        
    def getTargets(self, batch):
        batchLen = len(batch)
        
        states = np.array([ o[1][0] for o in batch ])
        states_ = np.array([ (self.nullState if o[1][3] is None else o[1][3]) for o in batch ])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)
        tp_ = self.tBrain.predict(states_)
        
        if self.debug:
            self.logs['cnt'] += 1
            if self.logs['cnt'] == 10:
                self.logs['batch'] = batch
                self.logs['states'] = states
                self.logs['states_'] = states_
                self.logs['p'] = np.copy(p)
                self.logs['p_'] = np.copy(p_)
                self.logs['tp_'] = np.copy(tp_)
            
        x = None
        if type(self.stateCnt) is tuple:
            x = np.zeros((batchLen, *self.stateCnt[0:len(self.stateCnt)]))
        else:
            x = np.zeros((batchLen, self.stateCnt))
        
        y = np.zeros((batchLen, self.actionCnt))
        errors = np.zeros(batchLen)

        for i in range(batchLen):
            o = batch[i][1]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            oldVal = t[a]
            if s_ is None:
                t[a] += (r - t[a]) * self.alpha
            else:
                t[a] += (max(-1, min(1, r + GAMMA * tp_[i][np.argmax(p_[i])])) - t[a]) * self.alpha

            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])
        
        if self.debug and self.logs['cnt'] == 10:
            self.logs['p_corrected'] = y

        return (x, y, errors)
            
    def updateTargetBrain(self):
        if self.debug: return
        self.brain.save()
        self.tBrain.set_weights(self.brain.get_weights())
        
    def filterIllMoves(self, moves, illMoves):
        for index, move in enumerate(moves):
            if index in illMoves:
                moves[index] = float("-inf")
        
        return moves
    
    def initLog(self):
        self.logs = {}
        self.logs['preds' + str(self.name)] = np.empty([0, self.actionCnt])
        self.logs['moves' + str(self.name)] = []
        self.logs['cnt'] = 0

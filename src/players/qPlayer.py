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
PLOT_INTERVAL = UPDATE_TARGET_FREQUENCY/5
BATCH_SIZE = 64

class QPlayer(Player):
    
    def __init__(self, name, game, **kwargs):
        super().__init__(name, game, **kwargs)
                
        self.nullState = np.zeros(self.stateCnt)
        self.verbosity = 0
        
        self.targetNet = kwargs['targetNet'] if 'targetNet' in kwargs else True
        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else BATCH_SIZE
        
        self.mem_cap = kwargs['mem_cap'] if 'mem_cap' in kwargs else MEMORY_CAPACITY

        self.memory = PMemory(self.mem_cap) if 'memory' not in kwargs else kwargs['memory']
        self.goodMemory = PMemory(self.mem_cap) if 'goodMemory' not in kwargs else kwargs['goodMemory']
        
        self.brain = kwargs['brain'] if 'brain' in kwargs else None
        self.tBrain = kwargs['tBrain'] if 'tBrain' in kwargs else None
        
        loadWeights = kwargs['loadWeights'] if 'loadWeights' in kwargs else False

        if self.brain is None:
            model = kwargs['model'] if "model" in kwargs else None
            self.brain = Brain(name, game, model=model)
            
            if self.targetNet:
                tModel = kwargs['tModel'] if model is not None else None
                self.tBrain = Brain(str(name) + "_target", game, model=tModel)
        
        if loadWeights:
            self.brain.load_weights()
        
        if self.eEq is None:
            self.eEq = MathEq({"min":MIN_EPSILON, "max":MAX_EPSILON, "lambda":E_LAMBDA})

        if self.aEq is None:
            self.aEq = MathEq({"min":MIN_ALPHA, "max":MAX_ALPHA, "lambda":A_LAMBDA})

        self.epsilon = self.eEq.getValue(0)
        self.alpha = self.aEq.getValue(0)

        self.initLog()
        if self.targetNet:
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
        
        self.sarsaMem.append(sample)
        self.updateR(sample[2])
        
        if game.isOver():
            if len(self.sarsaMem) < self.n_step: # if game ends before n steps
                self.increaseR()
            
            while len(self.sarsaMem) > 0:
                sample = self.getNSample(len(self.sarsaMem))
                self.addToReplayMemory(sample)
                self.R = (self.R - self.sarsaMem[0][2]) / self.gamma
                self.sarsaMem.pop(0)

            self.R = 0
            
            if self.targetNet and gameCnt % UPDATE_TARGET_FREQUENCY == 0:
                self.updateTargetBrain()

        if len(self.sarsaMem) >= self.n_step:
            sample = self.getNSample(len(self.sarsaMem))
            self.addToReplayMemory(sample)
            self.R = self.R - self.sarsaMem[0][2]
            self.sarsaMem.pop(0)
                
        self.verbosity = 2 if gameCnt % PLOT_INTERVAL == 0 and not game.isOver() else 0
        
    def train(self, game):
        batch = self.goodMemory.sample(int(self.batch_size/2))
        goodMemLen = len(batch)
        
        batch += self.memory.sample(int(self.batch_size - goodMemLen))
        
        if len(batch):
            x, y, errors = self.getTargets(batch)
        
        #update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            memory = self.goodMemory if i < goodMemLen else self.memory 
            memory.update(idx, errors[i])
        
        self.memory.releaseLock()
        self.goodMemory.releaseLock()
        
        if len(batch):
            self.brain.train(x, y, self.batch_size, self.verbosity)

            if self.debug:
                self.logs['x' + str(self.name)] = x
                self.logs['y' + str(self.name)] = y
        
    def addToReplayMemory(self, sample):
        _, _, errors = self.getTargets([(0, sample)])
        memory = self.goodMemory if sample[2] > 0 else self.memory
        memory.add(errors[0], sample)
        
    def getTargets(self, batch):
        batchLen = len(batch)
        
        states = np.array([ o[1][0] for o in batch ])
        states_ = np.array([ (self.nullState if o[1][3] is None else o[1][3]) for o in batch ])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)
        if self.targetNet:
            tp_ = self.tBrain.predict(states_)
        
        if self.debug:
            self.logs['cnt'] += 1
            if self.logs['cnt'] == 10:
                self.logs['batch'] = batch
                self.logs['states'] = states
                self.logs['states_'] = states_
                self.logs['p'] = np.copy(p)
                self.logs['p_'] = np.copy(p_)
                if self.targetNet:
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
                if self.targetNet:
                    t[a] += (max(-1, min(1, r + self.gamma_n * tp_[i][np.argmax(p_[i])])) - t[a]) * self.alpha
                else:
                    t[a] += (max(-1, min(1, r + self.gamma_n * p_[i][np.argmax(p_[i])])) - t[a]) * self.alpha

            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])
        
        if self.debug and self.logs['cnt'] == 10:
            self.logs['p_corrected'] = y

        return (x, y, errors)
            
    def updateTargetBrain(self):
        if self.debug: return
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

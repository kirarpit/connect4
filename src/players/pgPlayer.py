#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:52:18 2018

@author: Arpit
"""
import numpy as np
from mathEq import MathEq
from players.player import Player

class PGPlayer(Player):
    
    def __init__(self, name, game, **kwargs):
        super().__init__(name, game, **kwargs)
        
        self.sampling = kwargs['sampling'] if "sampling" in kwargs else True
        
        if 'brain' in kwargs:
            self.brain = kwargs['brain']
        else:
            print("Error: All policy gradient workers requrie a master brain")
            
        self.initLog()
        if self.eEq is None:
            self.eEq = MathEq({"min":0.05, "max":1, "lambda":0.001})
        
    def act(self, game):
        state = game.getCurrentState()
        illActions = game.getIllMoves()

        if np.random.uniform() < self.epsilon:
            action = self.getRandomMove(illActions)
        else:
            actions = self.brain.predict_p(np.array([state]))[0]
            fActions = self.filterIllMoves(np.copy(actions), illActions)
            if self.sampling:
                action = np.random.choice(self.actionCnt, p=fActions)
            else:
                action = np.argmax(fActions)
            
            if self.debug:
                self.logs['preds' + str(self.name)] = np.vstack([self.logs['preds' + str(self.name)], actions])
                self.logs['moves' + str(self.name)].append(action)
                
        return action

    def observe(self, sample, game): # where sample is (s, a, r, s_)
        super().observe(game)
        
        a_cats = np.zeros(self.actionCnt)	# turn action into one-hot representation
        a_cats[sample[1]] = 1
        
        self.sarsaMem.append((sample[0], a_cats, sample[2], sample[3]))
        self.updateR(sample[2])
        
        if game.isOver():
            if len(self.sarsaMem) < self.n_step: # if game ends before n steps
                self.increaseR()

            while len(self.sarsaMem) > 0:
                self.brain.train_push(self.getNSample(len(self.sarsaMem)))
                self.R = (self.R - self.sarsaMem[0][2]) / self.gamma
                self.sarsaMem.pop(0)
                
            self.R = 0
            
        if len(self.sarsaMem) >= self.n_step:
            self.brain.train_push(self.getNSample(self.n_step))
            self.R = self.R - self.sarsaMem[0][2]
            self.sarsaMem.pop(0)
        
    def train(self):
        pass
    
    def filterIllMoves(self, moves, illMoves):
        for index, move in enumerate(moves):
            if index in illMoves:
                moves[index] = 0 #since this time it's probabilities
            else:
                moves[index] += 1e-5 # in case all legal moves are zero
        
        moves /= moves.sum() #normalize
        return moves
    
    def initLog(self):
        self.logs = {}
        self.logs['preds' + str(self.name)] = np.empty([0, self.actionCnt])
        self.logs['moves' + str(self.name)] = []
        self.logs['cnt'] = 0

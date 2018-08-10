#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 14:36:06 2018

@author: Arpit
"""
from players.player import Player
from brains.zeroBrain import ZeroBrain
import numpy as np
from random import shuffle

class ZeroPlayer(Player):
    def __init__(self, name, game, **kwargs):
        super().__init__(name, game, **kwargs)
        
        self.tree = kwargs['tree']
        self.simCnt = kwargs["simCnt"] if "simCnt" in kwargs else 100
        self.iterCnt = kwargs['iterCnt'] if "iterCnt" in kwargs else 100
        self.tau = kwargs['tau'] if "tau" in kwargs else 1
        self.cpuct = kwargs['cpuct'] if "cpuct" in kwargs else 1
        self.mem = []
        self.gameMem = []

        model = kwargs['model'] if "model" in kwargs else None
        self.brain = ZeroBrain(name, game, model=model)
        if self.load_weights: self.brain.load_weights()

    def act(self, game):
        s = game.getCurrentState()
        
        game.save()
        for i in range(self.simCnt):
            self.MCTS(game, s)
            game.load()
            
        pi = [pow(self.tree['N'][(tuple(s), move)], 1.0/self.tau)
               if move not in game.getIllMoves() and (tuple(s), move) in self.tree['N']
               else 0 for move in range(game.actionCnt)]
        pi = [prob/sum(pi) for prob in pi]
        
        self.gameMem.append((s, pi, None))
        action = np.random.choice(game.actionCnt, p=pi)
        return action
    
    def observe(self, sample, game):
        if game.isOver():
            r = sample[2]
            self.gameMem = [(mem[0], mem[1], r) for mem in self.gameMem]
    
    def train(self, game):
        if game.isOver():
            self.mem += self.gameMem
            self.gameMem = []

            if game.gameCnt % self.iterCnt == 0:
                shuffle(self.mem)
                self.brain.train(self.mem)
                self.mem = []
                
            self.tree.flushDicts()
    
    def MCTS(self, game, state):
        s = tuple(state)
        
        if game.isOver():
            return game.getReward(game.toPlay)
            
        if s not in self.tree['P']:
            P, V = self.brain.predict(np.array([state]))
            self.tree['P'][s] = P
            self.tree['V'][s] = V
            self.tree['N'][s] = 1
            return V
        
        bestU = float("-inf")
        for a in range(game.actionCnt):
            if a not in game.getIllMoves():
                if (s, a) in self.tree['Q']:
                    U = self.tree['Q'][(s,a)] + self.cpuct * self.tree['P'][s][a] * pow(self.tree['N'][s], 0.5)/(1 + self.tree['N'][(s,a)])
                else:
                    U = self.cpuct * self.tree['P'][s][a] * pow(self.tree['N'][s] + 1e-8, 0.5)
                    
                if U > bestU:
                    bestU = U
                    bestAction = a
                    
        a = bestAction
        game.step(a)
        V = -1 * self.MCTS(game, game.getCurrentState())
        
        if (s, a) in self.tree['Q']:
            self.tree['Q'][(s, a)] = (self.tree['Q'][(s, a)] * 
                     self.tree['N'][(s, a)] + V)/(self.tree['N'][(s, a)] + 1)
            self.tree['N'][(s, a)] += 1
        else:
            self.tree['Q'][(s, a)] = V
            self.tree['N'][(s, a)] = 1

        self.tree['N'][s] += 1
        return V
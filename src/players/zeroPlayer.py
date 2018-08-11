#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 14:36:06 2018

@author: Arpit
"""
from players.player import Player
from brains.zeroBrain import ZeroBrain
import numpy as np
from timer_cm import Timer
from collections import deque
import random

class ZeroPlayer(Player):
    def __init__(self, name, game, **kwargs):
        super().__init__(name, game, **kwargs)
        
        self.tree = kwargs['tree']
        self.simCnt = kwargs["simCnt"] if "simCnt" in kwargs else 100
        self.perIter = kwargs['perIter'] if "perIter" in kwargs else 100
        self.tau = kwargs['tau'] if "tau" in kwargs else 1
        self.cpuct = kwargs['cpuct'] if "cpuct" in kwargs else 1
        self.epsilon = kwargs['epsilon'] if "epsilon" in kwargs else 0.25
        self.alpha = kwargs['alpha'] if "alpha" in kwargs else 0.8
        self.longTermMem = deque(maxlen=self.mem_size if self.mem_size is not None else 10000)
        self.gameMem = []
        
        model = kwargs['model'] if "model" in kwargs else None
        self.brain = kwargs['brain'] if "brain" in kwargs else ZeroBrain(name, game, model=model)
        if self.load_weights: self.brain.load_weights()

    def act(self, game):
        s = game.getStateID()
        
        game.save()
        for i in range(self.simCnt):
            self.MCTS(game)
            game.load()
            
        pi = [pow(self.tree['N'][(s, move)], 1.0/self.tau)
               if move not in game.getIllMoves() and (s, move) in self.tree['N']
               else 0 for move in range(game.actionCnt)]
        pi = [prob/sum(pi) for prob in pi]
        
        self.gameMem.append((game.getCurrentState(), pi, None))
        action = np.random.choice(game.actionCnt, p=pi)
        return action
    
    def observe(self, sample, game):
        if game.isOver():
            r = sample[2]
            self.gameMem = [(mem[0], mem[1], r) for mem in self.gameMem]
    
    def train(self, game):
        if game.isOver():
            self.tree.flushDicts()
            self.longTermMem += self.gameMem
            self.gameMem = []

            if game.gameCnt % self.perIter == 0:
                minibatch = random.sample(self.longTermMem, min(2048, len(self.longTermMem)))
                self.brain.train(minibatch)
    
    def MCTS(self, game):
        s = game.getStateID()
        edges = []
        
        addNoiseFlag = True
        while s in self.tree['P'] and not game.isOver():
            self.tree['N'][s] += 1
            
            epsilon = 0
            nu = [0] * game.actionCnt
            if addNoiseFlag:
                epsilon = self.epsilon
                nu = np.random.dirichlet([self.alpha] * game.actionCnt)
                addNoiseFlag = False
            
            bestUCB = float("-inf")
            actions = [a for a in range(game.actionCnt) if a not in game.getIllMoves()]
            for idx, a in enumerate(actions):
                noisyP = (1-epsilon)*self.tree['P'][s][a] + epsilon*nu[idx]
                U = self.cpuct * (noisyP) * np.sqrt(self.tree['N'][s])
                Q = 0
                
                if (s, a) in self.tree['Q']:
                    U = U/(1 + self.tree['N'][(s,a)])
                    Q = self.tree['Q'][(s,a)]
                    
                UCB = U + Q
                if UCB > bestUCB:
                    bestUCB = UCB
                    bestAction = a
            
            a = bestAction
            edges.append((s, a))
            game.step(a)
            s = game.getStateID()
                
        if game.isOver():
            V = game.getReward(game.toPlay)
            
        elif s not in self.tree['P']:
            P, V = self.brain.predict(np.array([game.getCurrentState()]))
            self.tree['P'][s] = P
            self.tree['V'][s] = V
            self.tree['N'][s] = 1
       
        for edge in reversed(edges):
            V *= -1
    
            if edge in self.tree['Q']:
                self.tree['Q'][edge] = (self.tree['Q'][edge] * 
                         self.tree['N'][edge] + V)/(self.tree['N'][edge] + 1)
                self.tree['N'][edge] += 1
            else:
                self.tree['Q'][edge] = V
                self.tree['N'][edge] = 1
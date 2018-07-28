#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 15:52:53 2018

@author: Arpit
"""
from players.player import Player
import games.c4Solver as C4Solver
import numpy as np
from mathEq import MathEq

class MinimaxC4Player(Player):
    
    def __init__(self, name, stateCnt, actionCnt, debug=False):
        super().__init__(name, stateCnt, actionCnt, debug)
        self.epsilon = 0.75 if debug else 1
        self.solver = C4Solver
        self.eq = MathEq(2)
    
    def act(self, game):
        illActions = game.getIllMoves()

        if np.random.uniform() < self.epsilon:
            action = self.getRandomMove(illActions)
        else:
            if len(illActions) == self.actionCnt - 1:#only one legal move left
                action = list(set(range(self.actionCnt)) - set(illActions))[0]
            else:
                action = self.solver.solve(game)

        return action
    
    def observe(self, sample, game):
        if sample[3] is None:
            if not self.debug:
                self.epsilon = self.eq.getValue(game.gameCnt)
        
    def train(self):
        pass
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 16:53:32 2018

@author: Arpit
"""
from players.player import Player
from games.t3MinMax import TicTacToeBrain as T3M2
import numpy as np
from mathEq import MathEq
from functools import lru_cache

class MinimaxT3Player(Player):
    
    def __init__(self, name, stateCnt, actionCnt, debug=False):
        super().__init__(name, stateCnt, actionCnt, debug)
        self.epsilon = 0.75 if debug else 1
        self.solver = T3M2()
        self.eq = MathEq(2)
    
    def act(self, game):
        illActions = game.getIllMoves()

        if np.random.uniform() < self.epsilon:
            action = self.getRandomMove(illActions)
        else:
            action = self.getBestMove(game.toString())

        return action
    
    def observe(self, sample, game):
        if sample[3] is None:
            if not self.debug:
                self.epsilon = self.eq.getValue(game.gameCnt)
        
    def train(self):
        pass
    
    @lru_cache(maxsize=None)
    def getBestMove(self, gameStr):
        self.solver.createBoard()
                
        for pos, move in enumerate(gameStr):
            if move != "0":
                player = "o" if move == "1" else "x"
                self.solver.makeMove(pos, player)
        
        return self.solver.minimax("x")[1]
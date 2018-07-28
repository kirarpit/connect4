#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 16:53:32 2018

@author: Arpit
"""
from players.player import Player
from games.t3MinMax import TicTacToeBrain as T3M2
from functools import lru_cache
import numpy as np

class MinimaxT3Player(Player):
    
    def __init__(self, name, game, **kwargs):
        super().__init__(name, game, **kwargs)
        self.solver = T3M2()
    
    def act(self, game):
        if np.random.uniform() < self.epsilon:
            action = self.getRandomMove(game.getIllMoves())
        else:
            action = self.getBestMove(game.toString())
        
        return action
    
    def observe(self, sample, game):
        super().observe(game)
        
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
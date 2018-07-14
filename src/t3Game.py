#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 14:16:25 2018

@author: Arpit
"""

from game import Game
from t3MinMax import TicTacToeBrain as T3M2
from functools import lru_cache
import numpy as np

class T3Game(Game):
    WINNER_R = 0.5
    LOSER_R = -1
    DRAW_R = 0
    
    def __init__(self, size=3):
        super().__init__()
        
        self.rows = 3
        self.columns = 3
        self.stateCnt = self.rows * self.columns * 2
        self.actionCnt = self.rows * self.columns
        
    def newGame(self):
        super().newGame()
        self.illMoves = set()
        
    def getNextState(self, action):
        self.step(action)

        if not self.isOver():
            self.p2act()
    
        newState = self.getCurrentState() if not self.isOver() else None
            
        return (newState, self.getReward(1))

    def p2act(self):
        if False and np.random.uniform() < 0.05:
            while True:
                action = np.random.choice(self.actionCnt, 1)[0]
                if action not in self.getIllMoves():
                    break
        else:
            action = self.getBestMove(self.toString())
            
        self.step(action)
        
    def step(self, action):
        super().step(action)
        
        x = int(action/self.rows)
        y = action % self.columns
        self.gameState[x][y] = self.toPlay
        self.updateArrayForm(x, y)
        self.illMoves.add(action)
        self.checkEndStates(x, y)
        self.switchTurn()
        
    def checkEndStates(self, row, column):
        if self.xInARow(row, column, 3):
            self.setWinner(self.toPlay)
            return
    
        self.checkDrawState()

    def getIllMoves(self):
        return list(self.illMoves)    
    
    @lru_cache(maxsize=None)
    def getBestMove(self, GS):
        game = T3M2()
        game.createBoard()
        
        pos = 0
        for x in range(0, self.rows):
            for y in range(0, self.columns):
                if self.gameState[x][y] != 0:
                    player = "o" if self.gameState[x][y] == 1 else "x"
                    game.makeMove(pos, player)
                pos += 1
        
        return game.minimax("x")[1]
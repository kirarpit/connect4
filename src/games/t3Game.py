#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 14:16:25 2018

@author: Arpit
"""
from games.game import Game

class T3Game(Game):
    WINNER_R = 1
    LOSER_R = -1
    DRAW_R = 0
    
    def __init__(self, size=3, isConv=False):
        super().__init__("T3", isConv)
        
        self.rows = 3
        self.columns = 3
        self.stateCnt = self.rows * self.columns * 2 if not self.isConv else (1, self.rows, self.columns)
        self.actionCnt = self.rows * self.columns
        
    def newGame(self):
        super().newGame()
        self.illMoves = set()
        
    def step(self, action):
        super().step(action)
        
        x = int(action/self.rows)
        y = action % self.columns
        self.gameState[x][y] = self.toPlay
        self.updateStateForm(x, y)
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
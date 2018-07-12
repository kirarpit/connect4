#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 14:16:25 2018

@author: Arpit
"""

from game import Game
import numpy as np

class T3Game(Game):
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
    
        if not self.isOver():
            newState = self.getCurrentState()
        else:
            newState = None
            
        return (newState, self.getReward(1))

    def p2act(self):
        while True:
            action = np.random.choice(self.actionCnt, 1)[0]
            if action not in self.getIllMoves():
                break
            
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
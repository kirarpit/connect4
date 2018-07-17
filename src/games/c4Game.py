#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 19:47:06 2018

@author: Arpit
"""

from games.game import Game
from games.c4Solver import C4Solver
import numpy as np
import random

class C4Game(Game):
    
    DRAW_R = -0.5

    def __init__(self, rows=6, columns=7):
        super().__init__("C4")
        
        self.rows = rows
        self.columns = columns
        self.stateCnt = (1, self.rows, self.columns)
        self.actionCnt = columns
        self.solver = C4Solver()

    def newGame(self):
        super().newGame()
        self.stateForm = np.zeros(self.stateCnt, dtype=np.uint8)
        self.stateForm[True] = 128
        
        self.columnString = ""
        self.fullColumns = set()
        
    def getNextState(self, action):
        self.step(action)
        
        if not self.isOver():
            self.p2act()
    
        if not self.isOver():
            newState = self.getCurrentState()
        else:
            newState = None
            
        return (newState, self.getReward(1))
        
    def step(self, column):
        super().step(column)
        
        row = 0
        while row < self.rows:
            if self.gameState[row][column] != 0:
                break
            row += 1
        
        row -= 1
        if row == 0:
            self.fullColumns.add(column)
            
        self.updateGameState(row, column)
    
    def updateGameState(self, row, column):
        self.gameState[row][column] = self.toPlay
        self.updateStateForm(row, column)
        self.columnString += str(column + 1)
        self.checkEndStates(row, column)
        self.switchTurn()
    
    def updateStateForm(self, row, column):
        if self.toPlay == 2:
            val = 64
        else:
            val = 192
        self.stateForm[0][row][column] = val
        
    def checkEndStates(self, row, column):
        if self.xInARow(row, column, 4):
            self.setWinner(self.toPlay)
            
        self.checkDrawState()
        
    def getIllMoves(self):
        return list(self.fullColumns)
        
    def p2act(self, epsilon=0.25):
        if np.random.uniform() < epsilon:
            while True:
                action = np.random.choice(self.actionCnt, 1)[0]
                if action not in self.getIllMoves():
                    break
        else:
            action = random.sample(self.solver.solve(self.columnString), 1)[0]
#            action = self.solver.solve(self.columnString)[0]

        self.step(action)
        
    def printGame(self):
        print ("#" * 19)
        super().printGame()
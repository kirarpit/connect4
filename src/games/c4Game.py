#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 19:47:06 2018

@author: Arpit
"""
from games.game import Game

class C4Game(Game):
    
    DRAW_R = 0.5

    def __init__(self, rows=6, columns=7):
        super().__init__("C4")
        
        self.rows = rows
        self.columns = columns
        self.stateCnt = rows * columns * 2
        self.actionCnt = columns

    def newGame(self):
        super().newGame()
        self.columnString = ""
        self.filledColumns = set()
        
    def step(self, column):
        if (super().step(column) < 0):
            print("Error!!!")
            print(self.columnString)
        
        row = 0
        while row < self.rows:
            if self.gameState[row][column] != 0:
                break
            row += 1
        
        row -= 1
        if row == 0:
            self.filledColumns.add(column)
            
        self.updateGameState(row, column)
    
    def updateGameState(self, row, column):
        self.gameState[row][column] = self.toPlay
        self.updateStateForm(row, column)
        self.columnString += str(column + 1)
        self.checkEndStates(row, column)
        self.switchTurn()
    
    def checkEndStates(self, row, column):
        if self.xInARow(row, column, 4):
            self.setWinner(self.toPlay)
            return
            
        self.checkDrawState()
        
    def checkDrawState(self):
        if super().checkDrawState():
            self.rewards[self.firstToPlay] = 0
            self.rewards[self.getNextPlayer(self.firstToPlay)] = self.DRAW_R
        
    def getIllMoves(self):
        return list(self.filledColumns)
                
    def printGame(self):
        print ("#" * 19)
        super().printGame()
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 19:47:06 2018

@author: Arpit
"""

from game import Game
import requests, yaml, random
from functools import lru_cache

LOSER_R = -5
WINNER_R = 5

@lru_cache(maxsize=None)
def getP2Move_1(gameString):
    r = requests.get('http://kevinalbs.com/connect4/back-end/index.php/getMoves?board_data='
                     + gameString + '&player=2')
    moves = yaml.safe_load(r.text)
    return int(max(moves, key=moves.get))

@lru_cache(maxsize=None)
def getP2Move_2(gameColumnString):
    r = requests.get('http://connect4.gamesolver.org/solve?pos=' + str(gameColumnString))
    data = yaml.safe_load(r.text)
    moves = data['score']
    
    choices = []
    maxi = float("-inf")
    for index, value in enumerate(moves):
        if value != 100 and maxi <= value:
            if maxi<value:
                choices = [index]
                maxi = value
            else:
                choices.append(index)
        
    return choices
    
class C4Game(Game):
    
    def __init__(self, rows=6, columns=7):
        super().__init__()
        
        self.rows = rows
        self.columns = columns
        self.stateCnt = rows * columns * 2
        self.actionCnt = columns

        self.p2DiffLevel = 5
    
    def newGame(self):
        super().newGame()
        
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
        self.updateArrayForm(row, column)
        self.columnString += str(column + 1)
        self.checkEndStates(row, column)
        self.switchTurn()
        
    def checkEndStates(self, row, column):
        if self.xInARow(row, column, 4):
            self.setWinner(self.toPlay)
            
        self.checkDrawState()
        
    def getIllMoves(self):
        return list(self.fullColumns)
        
    def p2act(self):
        if self.p2DiffLevel == 3:
            action = getP2Move_1(self.toString())
        elif self.p2DiffLevel == 5:
            action = random.sample(getP2Move_2(self.columnString), 1)[0]

        self.step(action)
        
    def printGame(self):
        print ("#" * 19)
        super().printGame()
        
    def toString(self):
        lStr = ""
        for x in range(0, self.rows):
            for y in range(0, self.columns):
                if self.gameState[x][y] == -1:
                    lStr += str(0)
                else:
                    lStr += str(self.gameState[x][y])
        return lStr
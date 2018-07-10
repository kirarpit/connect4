#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 19:47:06 2018

@author: Arpit
"""

import numpy as np
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
    
class C4Game:
    
    def __init__(self, rows=6, columns=7):
        self.rows = rows
        self.columns = columns
        self.stateCnt = rows * columns * 2
        self.actionCnt = columns
        self.gameCnt = 0
        self.deltas = {
                "N":(0, -1),
                "NE":(1, -1),
                "NW":(-1, -1),
                "E":(1, 0),
                "W":(-1, 0),
                "SE":(1, 1),
                "SW":(-1, 1),
                "S":(0, 1)
                }
        self.stats = {1:0, 2:0, 'Draw':0}
        self.p2DiffLevel = 5
    
    def newGame(self):
        self.over = False
        self.rewards = {}
        self.gameCnt += 1
        self.toPlay = 1
        self.turnCnt = 0
        self.arrayForm = np.zeros(self.stateCnt, dtype=int)
        self.arrayForm[True] = -1
        self.gameState = np.zeros((self.rows, self.columns), dtype=int)
        self.columnString = ""
        self.fullColumns = set()
        
    def getStateActionCnt(self):
        return (self.stateCnt, self.actionCnt)

    def isOver(self):
        return True if self.over else False
        
    def getCurrentState(self):
        return self.arrayForm
        
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
        if self.isOver():
            print ("Game's over already.")
            return -1

        if column in self.fullColumns:
            print ("Illegal Move!!!!")
            return -2
        
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

        pos = row * self.columns + column
        if self.toPlay == 2:
            pos += self.rows * self.columns
        self.arrayForm[pos] = 1
        
        self.columnString += str(column + 1)
        self.checkEndStates(row, column)
        self.switchTurn()
        
    def checkEndStates(self, row, column):
        player = self.toPlay
        directions = [('N', 'S'), ('E', 'W'), ('NE', 'SW'), ('SE', 'NW')]

        for dpair in directions:
            cnt = 1
            for direction in dpair:
                (dx, dy) = self.deltas[direction]
                x = column + dx
                y = row + dy
                
                while x>=0 and x<self.columns and y>=0 and y<self.rows:
                    if player == self.gameState[y][x]:
                        cnt += 1
                    else:
                        break

                    x += dx
                    y += dy
                
            if cnt >= 4:
                self.setWinner(player)
                break
            
        if self.turnCnt == self.rows * self.columns - 1:
            self.over = True
            self.stats['Draw'] += 1
    
    def getIllMoves(self):
        return list(self.fullColumns)
        
    def p2act(self):
        if self.p2DiffLevel == 3:
            action = getP2Move_1(self.toString())
        elif self.p2DiffLevel == 5:
            action = random.sample(getP2Move_2(self.columnString), 1)[0]

        self.step(action)
        
    def switchTurn(self):
        self.turnCnt += 1
        self.toPlay = self.getNextPlayer(self.toPlay)

    def printGame(self):
        print ("#" * 19)
        print ("Total Games Played: " + str(self.gameCnt))
        print ("Winner Stats: " + str(self.stats))
        print ("Game " + str(self.gameCnt) + ":")
        for x in range(0, self.rows):
            for y in range(0, self.columns):
                print (str(self.gameState[x][y]) + "  ", end='')
            print ("\n")
        print ("Winner: " + str(self.over))
        print ("No. of turns: " + str(self.turnCnt))
        
    def setWinner(self, player):
        self.over = player
        self.rewards[player] = WINNER_R
        self.rewards[self.getNextPlayer(player)] = LOSER_R
        self.stats[player] += 1
        
    def getReward(self, player):
        if player in self.rewards:
            return self.rewards[player]
        else:
            return 0
        
    def getNextPlayer(self, player):
        if player == 1:
            return 2
        else:
            return 1
        
    def toString(self):
        lStr = ""
        for x in range(0, self.rows):
            for y in range(0, self.columns):
                if self.gameState[x][y] == -1:
                    lStr += str(0)
                else:
                    lStr += str(self.gameState[x][y])
        return lStr
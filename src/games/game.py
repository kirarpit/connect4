#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 13:42:11 2018

@author: Arpit
"""

from abc import ABC, abstractmethod
from graphPlot import GraphPlot
import numpy as np

class Game(ABC):
    WINNER_R = 1
    LOSER_R = -1
    
    def __init__(self, name, isConv=False):
        self.name = name
        self.isConv = isConv
        self.gameCnt = 0
        self.initStats()
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
        self.directions = [('N', 'S'), ('E', 'W'), ('NE', 'SW'), ('SE', 'NW')]
        self.gPlot = GraphPlot(name, 1, 3, list(self.stats[1].keys()))
    
    @abstractmethod
    def newGame(self):
        self.over = False
        self.rewards = {}
        self.gameCnt += 1
        self.toPlay = 1
        self.firstToPlay = 1
        self.turnCnt = 0
        self.gameState = np.zeros((self.rows, self.columns), dtype=int)
        
        if self.isConv:
            self.stateForm = np.zeros(self.stateCnt, dtype=np.uint8)
            self.stateForm[True] = 128
        else:
            self.stateForm = np.zeros(self.stateCnt, dtype=float)
            self.stateForm[True] = 0.01

        if self.gameCnt % 1000 == 0: self.gPlot.save()
    
    @abstractmethod
    def step(self, action):
        if self.isOver():
            print ("Game's over already.")
            return -1

        if action in self.getIllMoves():
            print ("Illegal Move!!!!")
            print (self.gameState)
            print ("To play " + str(self.toPlay))
            print ("Wrong move " + str(action))
            return -2
    
        return 1
    
    @abstractmethod
    def getIllMoves(self):
        pass

    def xInARow(self, row, column, length):
        for dpair in self.directions:
            cnt = 1
            for direction in dpair:
                (dy, dx) = self.deltas[direction]
                x = row + dx
                y = column + dy
                
                while y>=0 and y<self.columns and x>=0 and x<self.rows:
                    if self.toPlay == self.gameState[x][y]:
                        cnt += 1
                    else:
                        break

                    x += dx
                    y += dy
                
            if cnt >= length:
                return cnt

        return False
    
    def setFirstToPlay(self, player):
        self.firstToPlay = player
        self.toPlay = self.firstToPlay
        
    def setWinner(self, player):
        self.over = player
        self.rewards[player] = self.WINNER_R
        self.rewards[self.getNextPlayer(player)] = self.LOSER_R
        self.stats[self.firstToPlay][player] += 1
        
    def updateStateForm(self, row, column):
        if self.isConv:
            if self.toPlay == 2:
                val = 64
            else:
                val = 192
            self.stateForm[0][row][column] = val
        else:
            pos = row * self.columns + column
            if self.toPlay == 2:
                pos += self.rows * self.columns
            self.stateForm[pos] = 1
        
    def checkDrawState(self):
        if self.turnCnt == self.rows * self.columns - 1:
            self.over = 3
            self.stats[self.firstToPlay]['Draw'] += 1
            self.rewards[self.toPlay] = self.DRAW_R
            self.rewards[self.getNextPlayer(self.toPlay)] = self.DRAW_R
            return True
        return False
    
    def getCurrentState(self):
        return np.copy(self.stateForm)
    
    def getStateActionCnt(self):
        return (self.stateCnt, self.actionCnt)
    
    def isOver(self):
        return True if self.over else False
    
    def switchTurn(self):
        self.turnCnt += 1
        self.toPlay = self.getNextPlayer(self.toPlay)

    def getNextPlayer(self, player):
        if player == 1:
            return 2
        else:
            return 1

    def getReward(self, player):
        if player in self.rewards:
            return self.rewards[player]
        else:
            return 0
        
    def clearStats(self):
        self.gPlot.add(self.gameCnt, np.add(list(self.stats[1].values()), list(self.stats[2].values())))
        self.initStats()

    def initStats(self):
        self.stats = {1:{1:0, 2:0, 'Draw':0}, 2:{1:0, 2:0, 'Draw':0}}
        
    def showGraph(self):
        self.gPlot.show()
        
    def toString(self):
        lStr = ""
        for x in range(0, self.rows):
            for y in range(0, self.columns):
                lStr += str(self.gameState[x][y])
        return lStr
    
    def printGame(self):
        print ("Total Games Played: " + str(self.gameCnt))
        print ("Winner Stats: " + str(self.stats))
        print ("Game " + str(self.gameCnt) + ":")
        for x in range(0, self.rows):
            for y in range(0, self.columns):
                print (str(self.gameState[x][y]) + "  ", end='')
            print ("\n")
        print ("Winner: " + str(self.over))
        print ("No. of turns: " + str(self.turnCnt))
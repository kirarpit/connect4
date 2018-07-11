#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 13:42:11 2018

@author: Arpit
"""

from abc import ABC, abstractmethod

class Game(ABC):
    def __init__(self):
        self.gameCnt = 0
    
    @abstractmethod
    def newGame(self):
        self.over = False
        self.rewards = {}
        self.gameCnt += 1
        self.toPlay = 1
        self.turnCnt = 0
        
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
        
    @abstractmethod
    def getCurrentState(self):
        pass
    
    @abstractmethod
    def getNextState(self, action):
        pass
    
    @abstractmethod
    def step(self, action):
        pass
    
    @abstractmethod
    def getIllMoves(self):
        pass
    
    @abstractmethod
    def printGame(self):
        pass
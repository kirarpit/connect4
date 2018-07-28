#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 14:29:37 2018

@author: Arpit
"""
from abc import ABC, abstractmethod
import time

class Environment(ABC):
    def __init__(self, game, debug):
        self.game = game
        self.debug = debug
        self.startTime = time.time()
    
    @abstractmethod
    def setPlayers(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    @abstractmethod
    def run(self):
        while not self.debug or self.game.gameCnt < 10:
            self.runGame()
            
            if self.game.gameCnt % 100 == 0 or self.debug:
                self.game.printGame()
                print ("Exploration Rate: " + str(self.p1.epsilon))
                print ("Learning Rate: " + str(self.p1.alpha))
                self.game.clearStats()
                print("Time since beginning: " + str(time.time() - self.startTime))

    def runGame(self):
        self.game.newGame()
        
        flag = 0
        if self.game.gameCnt % 2 == 0: # switch first to play alternatively
            self.game.setFirstToPlay(2)
            flag = 1

        lastS = None
        lastA = None
        while not self.game.isOver():
            p = self.p1 if (self.game.turnCnt + flag) % 2 == 0 else self.p2
            
            s = self.game.getCurrentState()
            a = p.act(self.game)
            self.game.step(a)
    
            if lastS is not None:
                self.teachLastPlayer(lastS, lastA, flag)
            
            lastS = s
            lastA = a
            
        self.game.turnCnt += 1
        self.teachLastPlayer(lastS, lastA, flag)

    def teachLastPlayer(self, lastS, lastA, flag): # if a player has played then previous turn player can get their rewards, observe sample and train
        p = self.p1 if (self.game.turnCnt + flag) % 2 == 0 else self.p2
        
        r = self.game.getReward(p.name)
        s_ = self.game.getCurrentState() if not self.game.isOver() else None
    
        sample = (lastS, lastA, r, s_)
        p.observe(sample, self.game)
        p.train()
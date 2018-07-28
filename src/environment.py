#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 14:29:37 2018

@author: Arpit
"""
from graphPlot import GraphPlot
import time

class Environment():
    def __init__(self, game, p1, p2, debug=False):
        self.game = game
        self.p1 = p1
        self.p2 = p2
        self.debug = debug
        
        self.startTime = time.time()
        self.gPlot = GraphPlot("e-rate-" + str(self.game.name), 1, 2, ["p1-e", "p2-e"])

    def run(self):
        while not self.debug or self.game.gameCnt < 10:
            self.runGame()
            
            if self.game.gameCnt % 100 == 0 or self.debug:
                self.printEnv()
                
            if self.game.gameCnt % 1000 == 0:
                self.gPlot.save()

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
        if not self.debug:
            p.train()
        
    def printEnv(self):
        self.game.printGame()
        print ("p1-e: " + str(self.p1.epsilon))
        print ("p2-e: " + str(self.p2.epsilon))
        print ("Learning Rate: " + str(self.p1.alpha))
        self.game.clearStats()
        print("Time since beginning: " + str(time.time() - self.startTime))
        print("#"*50)
              
        if not self.debug:
            self.gPlot.add(self.game.gameCnt, [self.p1.epsilon, self.p2.epsilon])

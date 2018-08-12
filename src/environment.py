#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 14:29:37 2018

@author: Arpit
"""
from graphPlot import GraphPlot
import time, os
from timer_cm import Timer
from evaluator import Evaluator

class Environment():
    def __init__(self, game, p1, p2, **kwargs):
        self.startTime = time.time()

        self.game = game
        self.p1 = p1
        self.p2 = p2
        self.debug = kwargs['debug'] if "debug" in kwargs else False
        self.training = kwargs['training'] if "training" in kwargs else True
        self.observing = kwargs['observing'] if "observing" in kwargs else True
        self.thread = kwargs['thread'] if "thread" in kwargs else False
        self.ePlotFlag = kwargs['ePlotFlag'] if "ePlotFlag" in kwargs else False
        self.gPlotFlag = kwargs['gPlotFlag'] if "gPlotFlag" in kwargs else True
        self.switchFTP = kwargs['switchFTP'] if "switchFTP" in kwargs else True
        self.evaluate = kwargs['evaluate'] if "evaluate" in kwargs else False
        if self.evaluate:
            self.evaluator = Evaluator(self.game)
        
        if self.ePlotFlag:
            self.ePlot = GraphPlot("e-rate-" + str(self.game.name), 1, 2, ["p1-e", "p2-e"])
        if self.gPlotFlag:
            self.gPlot = GraphPlot("game-stats-" + str(self.game.name), 1, 3, list(self.game.stats[1].keys()))

    def run(self):
        while not self.debug or self.game.gameCnt < 10:
            self.runGame()

            if self.game.gameCnt % 100 == 0 or self.debug:
                self.printEnv()
            
            if self.game.gameCnt % 100 == 0:
                if self.ePlotFlag: self.ePlot.save()
                if self.gPlotFlag: self.gPlot.save()
                self.p1.brain.save()
                    
                newModel = self.p1.brain.name
                oldModel = str(newModel) + "_old"
                if self.evaluate and os.path.exists(oldModel + ".h5"):
                    result = self.evaluator.evaluate(newModel, oldModel)
                    if not result:
                        print("New model is discarded!!!!!!!!!!!!!!!")
                        self.p1.brain.load_weights(oldModel)
                    else:
                        print("New model is good!!!!!!!!!!!!!!!")
                        
                self.p1.brain.save(oldModel)

            if self.thread: break
        
    def runGame(self):
        self.game.newGame()
        
        if self.switchFTP and self.game.gameCnt % 2 == 0: # switch first to play alternatively
            self.game.setFirstToPlay(2)

        lastS = None
        lastA = None
        while not self.game.isOver():
            p = self.p1 if self.game.toPlay == 1 else self.p2
            
            s = self.game.getCurrentState()
            a = p.act(self.game)
            self.game.step(a)
    
            if lastS is not None:
                self.teachLastPlayer(lastS, lastA)
            
            lastS = s
            lastA = a
            
        self.game.switchTurn() # Dummy switch so that the player who made the last action could take the reward 
        self.teachLastPlayer(lastS, lastA)

    def teachLastPlayer(self, lastS, lastA): # if a player has played then previous turn player can get their rewards, observe sample and train
        p = self.p1 if self.game.toPlay == 1 else self.p2
        
        r = self.game.getReward(self.game.toPlay)
        s_ = self.game.getCurrentState() if not self.game.isOver() else None
    
        sample = (lastS, lastA, r, s_)
        
        if self.observing:
            p.observe(sample, self.game)
        if self.training:
            p.train(self.game)
        
    def printEnv(self):
        self.game.printGame()
        print ("p1-e: " + str(self.p1.epsilon))
        print ("p2-e: " + str(self.p2.epsilon))
        if self.p1.alpha is not None:
            print ("Learning Rate: " + str(self.p1.alpha))
        print("Time since beginning: " + str(time.time() - self.startTime))
        print("#"*50)
              
        if not self.debug:
            if self.ePlotFlag: self.ePlot.add(self.game.gameCnt, [self.p1.epsilon, self.p2.epsilon])
            if self.gPlotFlag: self.gPlot.add(self.game.gameCnt, self.game.getTotalWins())

        self.game.clearStats()
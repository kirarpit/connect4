#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 15:47:32 2018

@author: Arpit
"""
from abc import ABC, abstractmethod
import numpy as np

class Player(ABC):
    
    def __init__(self, name, game, **kwargs):
        self.name = name
        self.stateCnt, self.actionCnt = game.getStateActionCnt()
        
        self.debug = kwargs['debug'] if "debug" in kwargs else False
        self.eEq = kwargs['eEq'] if "eEq" in kwargs else None
        self.aEq = kwargs['aEq'] if "aEq" in kwargs else None
        
        self.epsilon = 0
    
    @abstractmethod
    def act(self):
        pass
    
    @abstractmethod
    def observe(self, game):
        if game.isOver() and self.eEq is not None:
            if not self.debug:
                self.epsilon = self.eEq.getValue(game.gameCnt)
    
    @abstractmethod
    def train(self):
        pass
    
    def getRandomMove(self, illActions):
        while True:
            action = np.random.choice(self.actionCnt, 1)[0]
            if action not in illActions:
                break

        return action
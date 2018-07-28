#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 15:47:32 2018

@author: Arpit
"""
from abc import ABC, abstractmethod
import numpy as np

class Player(ABC):
    
    def __init__(self, name, stateCnt, actionCnt, debug):
        self.name = name
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.debug = debug
    
    @abstractmethod
    def act(self):
        pass
    
    @abstractmethod
    def observe(self):
        pass
    
    @abstractmethod
    def train(self):
        pass
    
    def getRandomMove(self, illActions):
        while True:
            action = np.random.choice(self.actionCnt, 1)[0]
            if action not in illActions:
                break

        return action
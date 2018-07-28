#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 16:43:06 2018

@author: Arpit
"""
from envs.environment import Environment
from players.qPlayer import QPlayer

class SelfPlayEnv(Environment):
    def __init__(self, game, debug):
        super().__init__(game, debug)
        stateCnt, actionCnt = self.game.getStateActionCnt()
        p1 = QPlayer(1, stateCnt, actionCnt, debug)
        p2 = QPlayer(2, stateCnt, actionCnt, debug)
        self.setPlayers(p1, p2)

    def setPlayers(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
    
    def run(self):
        super().run()
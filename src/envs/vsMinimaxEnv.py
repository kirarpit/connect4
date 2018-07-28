#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 15:04:44 2018

@author: Arpit
"""
from envs.environment import Environment
from players.qPlayer import QPlayer

class VSMinimaxEnv(Environment):
    def __init__(self, game, debug):
        super().__init__(game, debug)
        stateCnt, actionCnt = self.game.getStateActionCnt()
        p1 = QPlayer(1, stateCnt, actionCnt, debug)
        
        if game.name == "T3":
            from players.minimaxT3Player import MinimaxT3Player
            p2 = MinimaxT3Player(2, stateCnt, actionCnt, debug)
        else:
            from players.minimaxC4Player import MinimaxC4Player
            p2 = MinimaxC4Player(2, stateCnt, actionCnt, debug)
            
        self.setPlayers(p1, p2)

    def setPlayers(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
    
    def run(self):
        super().run()
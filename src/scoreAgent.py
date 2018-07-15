#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:55:37 2018

@author: Arpit
"""

from games.t3Game import T3Game
from player import Player

game = T3Game()
stateCnt, actionCnt = game.getStateActionCnt()
p1 = Player("1-20", stateCnt, actionCnt, True)

while game.gameCnt < 5000:
    game.newGame()
    
    while not game.isOver():
        s = game.getCurrentState()
        a = p1.act(s, game.getIllMoves())
        game.getNextState(a)

game.printGame()
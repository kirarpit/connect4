#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 18:50:06 2018

@author: Arpit
"""
import requests, yaml, os
from cache import cached

def solve(gameState):
    return miniMax4X5Shell(gameState)

@cached
def miniMax6X7Kevin(gameString):
    r = requests.get('http://kevinalbs.com/connect4/back-end/index.php/getMoves?board_data='
                     + gameString + '&player=2')
    moves = yaml.safe_load(r.text)
    return int(max(moves, key=moves.get))

@cached
def miniMax4X5Shell(gameColumnString):
    if gameColumnString == "":
        gameColumnString = "null"
    
    moves = os.popen("./games/c4solver45 " + str(gameColumnString)).read().split()
    moves = list(map(int, moves))
    moves = [i * -1 for i in moves]
    
    return getBestMoves(moves)
    
@cached
def miniMax4X5API(self, gameColumnString):
    r = requests.get('http://connect4.gamesolver.org/solve?pos=' + str(gameColumnString))
    data = yaml.safe_load(r.text)
    moves = data['score']

    return getBestMoves(moves)

def getBestMoves(moves):
    choices = []
    maxi = float("-inf")
    for index, value in enumerate(moves):
        if maxi <= value:
            if maxi<value:
                choices = [index]
                maxi = value
            else:
                choices.append(index)
        
    return choices
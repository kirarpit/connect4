#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 18:50:06 2018

@author: Arpit
"""
import requests, yaml, os, random
from cache import cached

debug = False

def solve(game):
    gameString = game.columnString

    if game.rows == 6 and game.columns == 7:
        moves = miniMax6X7Shell(gameString)    
    elif game.rows == 5 and game.columns == 6:
        moves = miniMax5X6Shell(gameString)
    elif game.rows == 4 and game.columns == 5:
        moves = miniMax4X5Shell(gameString)

#    action = random.sample(moves, 1)[0]
    action = moves[0]
    return action

@cached
def miniMax4X5Shell(gameString):
    if debug: print("miniMax4X5Shell called")
    scores = getScores("./games/c4solver45", gameString)
    return getBestMoves(scores)

@cached
def miniMax5X6Shell(gameString):
    if debug: print("miniMax5X6Shell called")
    scores = getScores("./games/c4solver56", gameString)
    return getBestMoves(scores)

@cached
def miniMax6X7Shell(gameString):
    if debug: print("miniMax6X7Shell called")
    if len(gameString) > 9:
        scores = getScores("./games/c4solver67", gameString)
    else:
        return miniMax6X7API(gameString)
    return getBestMoves(scores)

@cached
def miniMax6X7API(gameString):
    if debug: print("miniMax6X7API called")
    r = requests.get('http://connect4.gamesolver.org/solve?pos=' + str(gameString))
    data = yaml.safe_load(r.text)
    scores = [-99 if i == 100 else i for i in data['score']]
    if debug: print(scores)

    return getBestMoves(scores)

def getScores(solver, gameString):
    if gameString == "":
        gameString = "null"
    
    command = solver + " " + gameString
    if debug: print(command)

    moves = os.popen(command).read().split()
    moves = list(map(int, moves))
    moves = [i * -1 for i in moves]    
    
    if debug: print(moves)
    return moves
    
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
    
    if debug: print(choices)
    return choices
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 18:50:06 2018

@author: Arpit
"""
import requests, yaml, pickle, os.path

dictFN = "games/c4Solution.pickle"

class C4Solver:
    def __init__(self):
        self.cnt = 0
    
    def solve(self, gameState):
        self.cnt += 1
        
        if self.cnt % 10000 == 0:
            with open(dictFN, 'wb') as handle:
                pickle.dump(self.getP2Move_2.cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        return self.getP2Move_2(gameState)
    
    def cached(func):
        def wrapper(*args):
            try:
                return func.cache[args[1]]
            except KeyError:
                func.cache[args[1]] = result = func(args[1])
                return result 
            
        if os.path.exists(dictFN):
            print("Dict loaded")
            with open(dictFN, 'rb') as handle:
                wrapper.cache = func.cache = pickle.load(handle)
        else:
            wrapper.cache = func.cache = {}
        
        return wrapper

    def getP2Move_1(gameString):
        r = requests.get('http://kevinalbs.com/connect4/back-end/index.php/getMoves?board_data='
                         + gameString + '&player=2')
        moves = yaml.safe_load(r.text)
        return int(max(moves, key=moves.get))
    
    @cached
    def getP2Move_2(gameColumnString):
        r = requests.get('http://connect4.gamesolver.org/solve?pos=' + str(gameColumnString))
        data = yaml.safe_load(r.text)
        moves = data['score']
        
        choices = []
        maxi = float("-inf")
        for index, value in enumerate(moves):
            if value != 100 and maxi <= value:
                if maxi<value:
                    choices = [index]
                    maxi = value
                else:
                    choices.append(index)
            
        return choices
    
    def cacheInfo(self):
        print("Dict length: " + str(len(self.getP2Move_2.cache.keys())))
        print("Number of hits: " + str(self.cnt))
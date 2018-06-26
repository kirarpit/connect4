#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 18:46:51 2018

@author: Arpit
"""

# Importing the libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import game as c4game
import os.path

def create_ann():
    ann = Sequential()
    ann.add(Dense(units = 35, kernel_initializer = "uniform", activation = 'relu', input_dim = 84))
    ann.add(Dense(units = 20, kernel_initializer = "uniform", activation = 'relu'))
    ann.add(Dense(units = 7, kernel_initializer = "uniform", activation = 'softmax'))
    ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return ann

def save_config():
    json = ann.to_json()
    file = open("config.json", "w")
    file.write(json)
    file.close()
    
def test():
    game = c4game.Game(6, 7)
    game.dropDisc(3)
    game.dropDisc(4)
    game.dropDisc(3)
    game.dropDisc(4)
    game.dropDisc(3)
    ans = ann.predict(game.arrayForm)
    print np.argmax(ans)
    
def sample(probs):
    sum = 0.0
    r = np.random.uniform()
    for index, prob in enumerate(ans[0]):
        sum += prob
        if sum > r:
            return index

    return index

def get_rewards(winner):
    if winner == 1:
        p1 = 0.5
        p2 = -0.5
    elif winner == 2:
        p1 = -0.5
        p2 = 0.5
    else:
        p1 = 0
        p2 = 0
        
    return (p1, p2)

def normalize(x):
    x = x + abs(min(x))
    x = np.array([i/np.sum(x) for i in x])
    return x
    
ann = create_ann()

weights_file = "weights.hdf5"
if os.path.exists(weights_file):
    ann.load_weights(weights_file, by_name=False)

gameNo = 1
while gameNo <= 20000:
    game = c4game.Game(6, 7)

    X_train = game.arrayForm
    y_train = ann.predict(game.arrayForm)
    result = []

    while not (game.isOver()):
        
        ans = ann.predict(game.arrayForm)
        outcome = sample(ans)
#        outcome = np.argmax(ans)
    
        X_train = np.vstack([X_train, game.arrayForm])
        y_train = np.vstack([y_train, ans])
        result.append(outcome)

        if game.dropDisc(outcome) == -2:
            ans[0][outcome] -= 0.5
            ans[0] = normalize(ans[0])
            ann.fit(game.arrayForm, ans, verbose=0, epochs=5)
        
    X_train = np.delete(X_train, 0, 0)
    y_train = np.delete(y_train, 0, 0)
    
    ## promoting winner player moves and vice versa
    winner = game.isOver()
    (p1, p2) = get_rewards(winner)
    
    i = 0
    for row in y_train:
        if i%2 == 0:
            delta = p1
        else:
            delta = p2

        row[result[i]] += delta
        y_train[i] = normalize(row)
        i += 1
    
    ## training
    verbosity = 0
    if gameNo % 50 == 0:
        print "Game " + str(gameNo) + ":"
        game.printGameState()
        print "Winner: " + str(game.winner)
        verbosity = 2
    
    ann.fit(X_train, y_train, verbose=verbosity, batch_size=4)
    gameNo += 1
    
ann.save_weights(weights_file)
save_config()

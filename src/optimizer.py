#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 19:36:30 2018

@author: Arpit
"""
import threading

class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self, brain):
        self.brain = brain
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            self.brain.optimize()

    def stop(self):
        self.stop_signal = True

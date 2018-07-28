#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 17:29:15 2018

@author: Arpit
"""
import math

class MathEq:
    def __init__(self, params):
        self.type = params['type'] if "type" in params else 0

        if self.type == 1:
            self.eqNo = params['eqNo']
        else:
            self.min = params['min']
            self.max = params['max']
            self.rate = params['lambda']
        
    def getValue(self, x):
        if self.type == 1:
            if self.eqNo == 1:
                value = -0.8242347 + (0.99722 - -0.8242347)/(1 + (x/97103.41) ** 0.6462744)
            elif self.eqNo == 2:
                value = -0.2542789 + (0.9955262 - -0.2542789)/(1 + (x/50457.89) ** 0.9100507)
        else:
            value = self.min + (self.max - self.min) * math.exp(-1 * self.rate * x)
        
        return value
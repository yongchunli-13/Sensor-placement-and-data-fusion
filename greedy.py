#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## The implementation of the greedy algorithm

import user 
import math
import numpy as np
import datetime


# assign local names to functions in the user file
gen_data = user.gen_data
srankone = user.srankone
intgreedy = user.intgreedy

# Function grd needs input n and s; outputs the objectve value, 
# solution, matrix \sum_{i in S}v_i*v_i^T and its inverse, 
# and running time of the greedy algorithm
def grd(n, s): 
    c = 1
    x = [0]*n # chosen set
    y = [1]*n # unchosen set
    indexN = np.flatnonzero(y)
     
    gen_data(n) # load data
     
    index = 0
    X, Xs, fval = intgreedy(x)
    Y = np.zeros([n,n])
    Ys = np.zeros([n,n])
    val = fval
    
    start = datetime.datetime.now()
 
    while c < s+1:   
        Y, Ys,index,fval = srankone(X, Xs, indexN, n, val)   
        X = Y
        Xs = Ys
        val = fval         
        
        x[index] = 1
        y[index] = 0
        indexN = np.flatnonzero(y)    
        c = c + 1
        
    grdx = x # output solution of greedy
    grdf = val # output value of greedy
    
    end = datetime.datetime.now()
    time = (end - start).seconds 
    
    return grdf, grdx, X, Xs, time 


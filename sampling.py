#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## The implementation of the randomized sampling algorithm
import numpy as np
import frank_wolfe
import user
import random
import datetime
#import usermesp
import pandas as pd

# assign local names to functions in the user and frank_wolfe file
frankwolfe  = frank_wolfe.frankwolfe
f = user.f

# Function randrounding needs input \alpha-optimal solution of PC, n and s;
# outputs the solution, objective value and running time
def randrounding(xsol, indexN, n, s):
    np.random.seed(1)

    indexT = np.flatnonzero(xsol)   
    indexT = list(indexT)

    xsel=[]
    rxsol = [0]*n
    A=0.0
    B=0.0    
    A=calconvolve(xsel,xsol,indexT, n,s)
    j = 0
    for c in indexN:
        if sum(xsel) >=s:
            for i in range(n-c):
                xsel.append(0.0)
            break
        
        if j-sum(xsel)>=n-s:
            for i in range(s-int(sum(xsel))):
                rxsol[indexT[i]] = 1.0
            break
        
        num=random.uniform(0, 1)
        
        xsel.append(1)
        #print(c)
        indexT.remove(c)
        
        if len(indexT) <= 0:
            zerolen = int(s-sum(rxsol))
            inj = 0
            for ini in range(zerolen):
                if rxsol[inj] < 1:
                    rxsol[inj] = 1
                inj = inj + 1
            break
        else:           
            B=calconvolve(xsel,xsol, indexT, n,s)
        #print(xsol[c]*B/A-num)
        if((xsol[c]*B)/A >= num):
            A=B
            rxsol[c] = 1
        else:
            A=A - xsol[c]*B
            xsel[j]=0.0
            
        j = j+1
    #print(sum(rxsol))      
    return  rxsol, f(rxsol)

################################
def calconvolve(xsel,xsol, indexT, n,s):
    
    l=len(indexT)
    nz= sum(k>0 for k in xsel)
    #nz = sum(xsel)
    value=0.0

    acon=[1, xsol[indexT[0]]]
    for i in range(l-1):
        tempi = indexT[i+1]
        acon=np.convolve(acon,[1,xsol[tempi]])

    value=acon[s-nz]

    return value

# Sampling 
def sampling(n, s, N):
    start = datetime.datetime.now()
    # run Frank-Wolfe
    [xsol, primal, ftime]=frankwolfe(n, s)  
    indexN = np.flatnonzero(xsol) 
    print("The running time of Frank-Wolfe algorithm = ", ftime)
    print('The current objective value is:', primal)
    
    bestx = 0
    bestf = -1e+10
    for i in range(N):
        x,fval = randrounding(xsol, indexN, n, s)
        print(fval)
        if fval > bestf:         
            bestx = x
            bestf = fval
    
    end = datetime.datetime.now()
    time = (end-start).seconds
    
    return bestf, bestx, time

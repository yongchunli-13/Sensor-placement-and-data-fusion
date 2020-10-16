#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Implementation of Frank-Wolfe, local search and sampling Algorithms on the instance of n=124

import pandas as pd
import numpy as np
import local_search
import frank_wolfe
import sampling

# assign local names
localsearch  = local_search.localsearch
frankwolfe  = frank_wolfe.frankwolfe
sampling = sampling.sampling

# parameters
n = 2382 #number of non-reference buses

# Local Search Algorithm
loc = 0
df = pd.DataFrame(columns=('n', 's', 'objective value', 'time'))

for s in range(100,400,100): # set the values of s
    print("This is case ", loc+1)
    fval, xsol, time  = localsearch(n, s) 
    df.loc[loc] = np.array([n, s, fval, time])
    loc = loc+1  


# Frank-Wolfe Algorithm
loc = 0
df = pd.DataFrame(columns=('n', 's', 'upper bound', 'time'))

for s in range(100,300,100): # set the values of s
    print("This is case ", loc+1)
    x,  mindual, time  = frankwolfe(n, s) 
    df.loc[loc] = np.array([n, s,  mindual, time])
    loc = loc+1  

# Sampling Algorithm
loc = 0
df = pd.DataFrame(columns=('n', 's', 'objective value', 'time'))

N = 100 # the number of repetitions for sampling 
for s in range(100,400,100): # set the values of s
    print("This is case ", loc+1)
    fval, xsol, time  = sampling(n, s, N) 
    df.loc[loc] = np.array([n, s, fval, time])
    loc = loc+1  

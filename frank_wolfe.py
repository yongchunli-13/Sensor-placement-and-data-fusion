#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## The implementation of the Frank-Wolfe algorithm

import user
import numpy as np
import datetime
import local_search

# assign local names to functions in the user and local_search file
grad = user.grad_fw
gen_datamesp = user.gen_datamesp
gen_data = user.gen_data
localsearch  = local_search.localsearch


# Function frankwolfe needs input n, d and s; outputs the solution, 
# size of its support, upper bound and running time of the Frank-Wolfe algorithm
def frankwolfe(n, s): 
    
    start = datetime.datetime.now()
    
    # run local search
    Obj_f, x, ltime = localsearch(n, s)
    print("The running time of local search algorithm is: ", ltime)
    print('The current objective value is: ', Obj_f)
    
    gen_data(n)
    V, intval = gen_datamesp(n)
        
    gamma_t = 0.0  
    t = 0.0
    mindual = 1e+10
    dual_gap = 1 # duality gap
    Obj_f = 1 # primal value
    alpha = 1e-4 # target accuracy
    
    sel = np.flatnonzero(x) 
    fmat = 0.0
    for i in sel:
        fmat = fmat + V[i].T*V[i]
        
    abs_Obj_f = 1    
    while(dual_gap/abs_Obj_f > alpha):
        Obj_f, subgrad, y, dual_gap = grad(fmat, x, s)
        
        t = t + 1
        gamma_t = 2/(t+2) # step size
        
        x = [(1-gamma_t)*x_i for x_i in x] 
        y = [gamma_t*y_i for y_i in y]
        
        fmat = (1-gamma_t) * fmat
        sel = np.flatnonzero(y) 
           
        ymat = 0
        for i in sel:
            ymat = ymat + V[i].T*V[i]
            
        fmat = fmat + gamma_t*ymat
        
        x = np.add(x,y).tolist() # update x
        
        mindual = min(mindual, Obj_f+dual_gap) # update the upper bound
        print('dual gap =', dual_gap, 'objective value =', Obj_f)
        if Obj_f < 0:
            abs_Obj_f =  -Obj_f
        else:
            abs_Obj_f = Obj_f

    
    end = datetime.datetime.now()
    time = (end-start).seconds

    return x, mindual+intval, time
 

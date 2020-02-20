#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:59:07 2020

@author: xinshi
"""

from gurobipy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
from numpy.linalg import inv
import sys, os

#sys.path.append(os.path.dirname(os.path.realpath('__file__')) + '/benchmark')

import time



def get_expr_coos(expr, var_indices):
    for i in range(expr.size()):
        dvar = expr.getVar(i)
        yield expr.getCoeff(i), var_indices[dvar]
        
def get_matrix_coos(m):
    dvars = m.getVars()
    constrs = m.getConstrs()
    var_indices = {v: i for i, v in enumerate(dvars)}
    for row_idx, constr in enumerate(constrs):
        for coeff, col_idx in get_expr_coos(m.getRow(constr), var_indices):
            yield row_idx, col_idx, coeff
            
            



def FR_simplex(A, b, c, base, AIbase, tol, dinv, dprint):
    
    
    m = A.shape[0]
    n = A.shape[1]
    
    
    nonbase = list(set(range(n)).difference(set(base)))
    
    
    nonAIbase = list(set(range(n)).difference(set(AIbase)))


    ################## two-phase revised simplex algorithm ###########3
    
    bmatrix_inv = inv(A[:, base])
    bmatrix_inv[abs(bmatrix_inv) < tol]= 0.0
    
    check = np.dot(np.dot(np.transpose(c[base]), bmatrix_inv), A[:,nonbase]) - c[nonbase]
    check[abs(check) < tol] = 0.0
    
    
    print("solution: ")
    print(np.dot(np.dot(c[base], bmatrix_inv), b))
    
    count = 0
    
    flag = 0
    while max(check) > 0 and count < 40000:
        
        count += 1
    
        
        invar = nonbase[np.argmax(check)]
        
        # determine the out variable
        denomin = np.dot(bmatrix_inv, A[:, invar])
        denomin[abs(denomin) < tol] = 0.0
        if max(denomin) <= 0:
            print("unbounded")
            break
        
        numerator = np.dot(bmatrix_inv, b)
        
        numerator[abs(numerator) < tol] = 0.0
        
        candidate = np.zeros(m)
        for i in range(m):
            if denomin[i] > 0:
                candidate[i] = numerator[i]/float(denomin[i])
            else:
                candidate[i] = np.float('inf')
        candidate[abs(candidate) < tol] = 0.0
        outvar = base[np.argmin(candidate)]
        
        #####
        # replace outvar with invar
        base[base.index(outvar)] = invar
        nonbase[nonbase.index(invar)] = outvar
        
    #    break
        ############  compute B inverse #####
        eta = -denomin/denomin[np.argmin(candidate)]
        eta[np.argmin(candidate)] = 1/denomin[np.argmin(candidate)]
        eta[abs(eta) < tol] = 0.0
        
        aux_matrix = np.eye(m)
        aux_matrix[:,base.index(invar)] = eta
        
        bmatrix_inv = np.dot(aux_matrix, bmatrix_inv)
        bmatrix_inv[abs(bmatrix_inv) < tol]= 0.0
    #    print(check)
        
        ########## reversion ############
        if count%dprint == 0 or count == 1:
            if count%dinv == 0:
                bmatrix_inv = inv(A[:, base])
                bmatrix_inv[abs(bmatrix_inv) < tol]= 0.0
    
            print("Iteration " +str(count) ) 
            print("------------obj. value: ")
            print(np.dot(np.dot(c[base], bmatrix_inv), b) )
            print("------------obj. value: ")
            print('\n')
    
    
        ########### check if going to phase 2 ##########3
        if flag == 0 and max(base) < n - len(AIbase ):
            print("Phase 2")
            flag = 1
            A = A[:,nonAIbase]
            c = c[nonAIbase]
            nonbase = list(set(nonbase).difference(set(AIbase)))
            bmatrix_inv = inv(A[:, base])
            bmatrix_inv[abs(bmatrix_inv) < tol]= 0.0
            
        check = np.dot(np.dot(np.transpose(c[base]), bmatrix_inv), A[:,nonbase]) - c[nonbase]
        check[abs(check) < tol] = 0.0 
    
    print("optimal solution: ", np.dot(np.dot(c[base], bmatrix_inv), b))       
    
    print("No. of iterations: ", count)
    
    return np.dot(bmatrix_inv,b), np.dot(np.dot(c[base], bmatrix_inv), b)

def mps_read_with_gurobi(path):
    
    model = read(path)
    
    
    dvars = model.getVars()
    n_vars = len(dvars)
    
    ###################### add bounds on variables #############
    for i in range(len(dvars)):
        lb = dvars[i].getAttr("LB")
        ub = dvars[i].getAttr("UB")
        if ub != float('inf'):
            model.addConstr(dvars[i] <= ub)
            
    
    LB = model.getAttr('LB', dvars)
    UB = model.getAttr('UB', dvars)
    
    model.update()
    
    ####################### make use of gurobi to get constraint matrix from mps file #########
    
    
    constrs = model.getConstrs()
    n_cons = len(constrs)
    
    obj_coeffs = model.getAttr('Obj', dvars)
    
    b = model.getAttr('RHS', constrs)
    
    var_index = {v: i for i, v in enumerate(dvars)}
    constr_index= {c: i for i, c in enumerate(constrs)}
    
    nzs = pd.DataFrame(get_matrix_coos(model), columns=['row_idx', 'col_idx', 'coeff'])
    #plt.scatter(nzs.col_idx, nzs.row_idx, marker='.', lw=0)
    
    
    A = scipy.sparse.csr_matrix( (np.transpose(nzs[['coeff']].values)[0],  ( np.transpose(nzs[['row_idx']].astype(int).values)[0],\
                                               np.transpose(nzs[['col_idx']].astype(int).values)[0])))
    A = A.toarray()
    
    cons_sense = model.getAttr("Sense", constrs)
    
    c = np.array(obj_coeffs)
    if model.ModelSense == -1:
        c = - c
        
    return A, b, c, cons_sense, LB, UB
    


def preprocess(A, b, c, cons_sense, LB, UB):
    
    b = b - np.dot(A, LB)
    #c = c  + np.dot(c, LB)
    
    M = 50*max(c)
    
    
    
    ##############  processing ####################
    
    
    m = A.shape[0]
    n = A.shape[1]
    
    base = []
    A_ = []
    c_ = []
    
    AIbase = []
    
    column_count = n
    
    for i in range(len(b)):
        if b[i] < 0:
            b[i] = -b[i]
            A[i] = -A[i]
            if cons_sense[i] == "<":
                cons_sense[i] == ">"
            if cons_sense[i] == ">":
                cons_sense[i] == "<"
    
    
    for i in range(len(cons_sense)):
        if cons_sense[i] == "<":
            base.append(column_count)
            A_.append(np.eye(1, m, i)[0])
            c_.append(0)
            column_count += 1
        elif cons_sense[i] == "=":
            base.append(column_count)
            A_.append(np.eye(1, m, i)[0])
            c_.append(M)
            column_count += 1
        else:
            A_.append(-np.eye(1, m, i)[0])
            column_count += 1
            c_.append(0)
            
            base.append(column_count)
            AIbase.append(column_count)
            A_.append(np.eye(1, m, i)[0])
            c_.append(M)
            column_count += 1
            
    A_ = np.array(A_)
    c_ = np.array(c_)
            
    A = np.concatenate((A, np.transpose(A_)), axis = 1)
    c = np.concatenate((c, c_))
    
    
    
    
    m = A.shape[0]
    n = A.shape[1]
    
    nonbase = list(set(range(n)).difference(set(base)))
    nonAIbase = list(set(range(n)).difference(set(AIbase)))
    
    
    ########## update A, b, c, base, AIbase#########3
    nAI = len(AIbase)
    A = np.concatenate((A[:,nonAIbase], A[:,AIbase]), axis = 1)
    c = np.concatenate((c[nonAIbase], c[AIbase]))
    
    newbase = []
    for i in range(len(base)):
        if base[i] in nonAIbase:
            newbase.append(nonAIbase.index(base[i]))
        else:
            newbase.append(n-nAI +AIbase.index(base[i]))
    base = newbase
    AIbase = list(range(n-nAI, n))
    return A, b, c, base, AIbase

if __name__ == '__main__':

#    path = "/Users/xinshi/Documents/job/huawei/simplex/benchmark/air04.mps"
    path = "/Users/xinshi/Documents/job/huawei/simplex/benchmark/csched010.mps"
    
    
    A, b, c, cons_sense, LB, UB = mps_read_with_gurobi(path)
    
    
    start_time = time.time()
    
   
    ############## algorithm begins #################
    tol = pow(10, -5)
    
    
    A, b, c, base, AIbase = preprocess(A, b, c, cons_sense, LB, UB)
    
    
    # two-phase revised simplex algorithm 
    xeval, feval = FR_simplex(A, b, c, base, AIbase, tol, 1, 1)
    print("--- %s seconds ---" % (time.time() - start_time))
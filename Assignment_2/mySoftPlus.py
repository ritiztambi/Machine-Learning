# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 11:39:19 2018

@author: Krishna Chaitanya Bandi & Ritiz Tambi
"""

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import time
import matplotlib.pyplot as plt

def fetchBatchData(xData, yData, k):
    
    numObs = xData.shape[0]
    if k == numObs:
        xBatch = xData
        yBatch = yData
    elif k == 1:
        randNum = np.random.randint(numObs)
        xBatch = xData[randNum, :, None]
        xBatch = xBatch.T
        yBatch = yData[randNum, :]
    else:
        _, xBatch, _, yBatch = train_test_split(xData, yData, 
                                                test_size = k / numObs,
                                                stratify = yData)
    return xBatch, yBatch


def sigmoidFunction(elem):
    
    return 1 / (1 + np.exp(-elem))

def computeGradient(x, y, lambdaVal, w, a):
    
    sumFirstTerm = 0    
    for row in range(len(x)):

        yrow = y[row]
        xrow = x[row]
        termrow = (1 - ( yrow*np.dot(xrow, w) ) ) / a
        sigmoidValrow = sigmoidFunction(termrow)
        sumFirstTerm = sumFirstTerm + sigmoidValrow*(-1*(yrow*xrow))
    
    firstTerm = sumFirstTerm/len(x)
    secondTerm = 2*lambdaVal*w
    
    return firstTerm + secondTerm

def computeCost(x, y, lambdaVal, w, a):
    
    sumFirstTerm = 0
    for rown in range(len(x)):
        yrow = y[rown]
        xrow = x[rown]
        temptermrow = (1 - ( yrow * np.dot(xrow, w) ) ) / a
        termrow = a*np.log(1 + np.exp(temptermrow))
        
        sumFirstTerm = sumFirstTerm + termrow
        
    firstTerm = sumFirstTerm/len(x)
    secondTerm = lambdaVal*(np.linalg.norm(w)**2)
    
    return firstTerm + secondTerm

def trainSoftplus(X, Y, w, numIters = 100, k = 20, lambdaVal = 100, eta = 1e-05, a = 1):
    
    costPrevious = 0
    costValueList = list()
    
    for iteration in range(0, numIters):
        eta = 0.001    
        newX, newY = fetchBatchData(X, Y, k)
        w = w - eta*(computeGradient(newX, newY, lambdaVal, w, a))
        tempCost = computeCost(X, Y, lambdaVal, w, a)
        
        if abs(tempCost - costPrevious) < 1e-06:
            break
        costPrevious = tempCost
        costValueList.append(tempCost)
    
    return costValueList
        
        
def mySoftPlus(filename, m, numRuns = 5):
    
    mnist = pd.read_csv(filename, header = None)
    mnist = np.array(mnist)
    
    X = mnist[:, 1:]
    X = (X - X.min()) / (X.max() - X.min()) # To avoid the large value of exponential passed into the log function while calculating cost
#    X = X / 255
    Y = mnist[:, 0]
    Y[Y == 1] = 1
    Y[Y == 3] = -1
    Y = Y[:, None]
    
    numObs, numFeatures = X.shape
    numIterations = len(X)
    lambdaValue = 10
    
    wInitial = np.zeros(numFeatures)
    w = wInitial
    
    max_cost = 0
    max_x = 0
    times = [0]*numRuns
    
    print("Batch_Size :",m)
    print("NumRuns:",numRuns)
    print("Training Softplus")
    
    for t in range(numRuns):
        
        print("Run :", t+1)
        start_time = time.time()
        costs =  trainSoftplus(X = X, Y = Y, w = w, numIters = numIterations, k = m, lambdaVal = lambdaValue)
        max_x = max(len(costs), max_x)
        max_c = max(costs)
        max_cost = max(max_cost, max_c)
        stop_time = time.time()
        time_taken = stop_time - start_time
        times[t] = time_taken
        plt.plot(costs)
    
    plt.ylabel('svm cost function')
    plt.xlabel('iterations')
    plt.title("Batch_size : {}".format(m))
#    plt.savefig("batch_{}".format(m))
    plt.show()
    
    avg = np.mean(times)
    std = np.std(times)
    print("Mean run-time:",avg)
    print("Std run-time:",std)
    print()
    print()
    

filename = "MNIST-13.csv"
for kVal in [1, 20, 200, 1000, 2000]:
    mySoftPlus(filename, kVal)


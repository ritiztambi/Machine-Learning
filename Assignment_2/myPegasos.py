# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 11:39:19 2018

@author: Krishna Chaitanya Bandi & Ritiz Tambi
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import math
import random
from sklearn.model_selection import train_test_split


def get_batch_data(X,Y,batch_size):
    n_samples = X.shape[0]
    if(batch_size == 1):
        random_index = np.random.randint(X.shape[0])
        batchX, batchY = X[random_index,:].reshape(-1,X.shape[1]), Y[random_index].reshape(-1,1)
    elif(batch_size == X.shape[0]):
        return X,Y
    else :
        _, batchX, _, batchY = train_test_split(X, Y, test_size=batch_size /n_samples,shuffle = True)
    return np.array(batchX), np.array(batchY)

def train_cost(data,k):
    lamda = 0.1
    n_samples,n_features = data.shape
    Y = data[:,0]
    X = data[:,1:]
    X = X/255
   
    ktot = n_samples
    costs = []
    cost_init = 0

    w = np.ones(n_features-1)* (1/(n_features*math.sqrt(lamda)))
    w_prev = w.reshape(-1,1)
    
    for i in range(ktot):
        ''' Create batches with equal sampling from both classes '''
        At,yt = get_batch_data(X,Y,k)
        yt = yt.reshape(-1,1)
        ''' Get At+ and yt+ sets'''
        At_plus = []
        yt_plus = []

        res = np.multiply(yt,np.dot(At,w_prev))


        
        for j in range(res.shape[0]):
            if(res[j] < 1):
                At_plus.append(At[j])
                yt_plus.append(yt[j])
        
        At_plus = np.array(At_plus)
        yt_plus = np.array(yt_plus)
        
        if(len(At_plus)<1):
            continue

        l_r = 1/(lamda*(i+1))
        w_half = (1-l_r*lamda)*w_prev + ((l_r/k)*np.sum(np.multiply(At_plus,yt_plus),axis=0)).reshape(-1,1)
        max_m = max(1.0,1/(math.sqrt(lamda)*np.linalg.norm(w_half)))
        w = w_half * max_m
        w_prev = w
#        print(w.shape)
        

        ''' Get Base Cost/Error '''
        loss = 1 - yt*(np.dot(At,w))
#        print("loss")
#        print(np.sum(loss[loss>0]))
#        print(loss.shape)
#        print("cost")

        cost = (lamda/2)*np.dot(w.T,w) + np.sum(loss[loss>0])/k
#        cost = cost/n_samples  # For better graph
        
        if(k!=1):
            if(abs(cost - cost_init )<1e-05 or i > 400):
    #            print("Printing i")
    #            print(i)
                break
        else:
            if(abs(cost - cost_init )<1e-05):
    #            print("Printing i")
    #            print(i)
                break
        cost_init = cost
        costs.append(cost.flatten().item())
            
       
    return costs



def myPegasos(filename,k,numruns):
    dataframe = pd.read_csv(filename,header=None);
    data = np.array(pd.DataFrame(dataframe))
    target = data[:,0]
    
    ''' Relabel Y as +1 and -1 '''
    classes = np.unique(target)
    label_1 = classes[0]
    label_2 = classes[1]
    data[target==label_1,0] = 1;
    data[target==label_2,0] = -1;
    
    times = [0]*numruns
    print("Batch_Size :",k)
    print("NumRuns:",numruns)
    print("Training Pegasos")
    
    max_cost = 0
    max_x = 0
    for t in range(numruns):
        print("Run :",t+1)
        start_time = time.time()
        costs =  train_cost(data,k)
        max_x = max(len(costs),max_x)
        max_c = max(costs)
        max_cost = max(max_cost,max_c)
        stop_time = time.time()
        time_taken = stop_time - start_time
        times[t] = time_taken
        plt.plot(costs)
        
        
    plt.ylabel('svm cost function')
    plt.xlabel('iterations')
    plt.title("Batch_size : {}".format(k))
#    plt.savefig("Batch_size : {}".format(k))
    plt.show()  
    
    avg = np.mean(times)
    std = np.std(times)
    print("Mean run-time:",avg)
    print("Std run-time:",std)
    print()
    print()
    
#if __name__ == '__main__':
#    k_list = [1,20,200,1000,2000]
#    num_runs = 5
#    for k in k_list:
#        myPegasos('MNIST-13.csv',k,num_runs)
        
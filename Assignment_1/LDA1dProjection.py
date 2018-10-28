import pandas as pd
import numpy as np
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

def S_Within(data):
    target = data[:,-1]
    data = data[:,:-1]
    dim = data.shape[1]
    S_W = np.zeros((dim,dim))
    classes = np.unique(target)
    for i in classes:
        mean_class = np.mean(data[target==i],axis=0) 
        S_class = np.matmul((data[target==i]-mean_class).T,(data[target==i]-mean_class))
        S_W += S_class
    return S_W

def S_Between(data):
    target = data[:,-1]
    data = data[:,:-1]
    dim = data.shape[1]
    S_B = np.zeros((dim,dim))
    classes = np.unique(target)
    mean_data = np.mean(data,axis=0);
    
    for i in classes:
        mean_class = np.mean(data[target==i], axis=0)
        S_B += data[target==i].shape[0] * np.outer(mean_class - mean_data, mean_class - mean_data)
    return S_B


def projection_vector(data):
    target = data[:,-1]
    classes = np.unique(target)
    S_W = S_Within(data)
    S_B = S_Between(data)
    temp = np.dot(np.linalg.pinv(S_W) , S_B )
    w, v = eigs(temp, k = len(classes)-1)
    return v


def LDA1dProjection(filename,num_crossval):
    dataframe = pd.read_csv(filename)
    data = np.array(pd.DataFrame(dataframe))
    target = data[:,-1]
    median = np.median(target)
    target_new = [0 if i<=median else 1 for i in target]
    data[:,-1] = target_new
    target = data[:,-1]    
    v = projection_vector(data)
    
    X = data[:,:-1]
    projected_data = np.dot(X,v)
    P_0 = projected_data[target==0]
    P_1 = projected_data[target==1]
    plt.hist([P_0.real,P_1.real],bins=20,histtype='bar')
    plt.ylabel('Number of Projections')
    plt.xlabel('Bins')
    plt.title('Histogram')
    plt.show()
    
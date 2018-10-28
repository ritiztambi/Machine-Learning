import pandas as pd
import numpy as np
import scipy
import math
from scipy.sparse.linalg import eigs

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
    S_W = S_Within(data)
    S_B = S_Between(data)
    temp = np.dot(np.linalg.pinv(S_W) , S_B )
    w, v = eigs(temp, k = 2)
    return v



def get_mu(data):
    X = data[:,:-1]
    Y = data[:,-1]
    classes = [0,1,2,3,4,5,6,7,8,9]
    n_c = 10
    f = X.shape[1]
    mu = np.zeros((n_c,f))
    for k in classes:
        X_class = X[Y==k, :]
        mu[k,:] = np.mean(X_class, axis=0)
    return mu
    
    

def get_sigma(data):
    X = data[:,:-1]
    Y = data[:,-1]
    n_c = 10
    classes = [0,1,2,3,4,5,6,7,8,9]
    f = X.shape[1]
    sigma = np.zeros((n_c,f,f))
    for k in classes:
        X_class = X[Y==k, :]
        sigma[k,:,:] = np.cov(X_class,bias=1,rowvar=0)
    return sigma
    

def get_prior(Y):
    
    classes = [0,1,2,3,4,5,6,7,8,9]
    len_data = Y.shape[0]
    prior = np.array([np.sum(Y == cls) / len_data for cls in classes])
    return prior



def predict_class(data,mu,sigma,prior):
    X = data[:,:-1]
    classes = [0,1,2,3,4,5,6,7,8,9]
    log_prior = np.array([math.log(p) for p in prior])
    n_c = 10
    scores = np.zeros((data.shape[0],n_c))
    for k in classes:
        mu_class = mu[k,:]
        sigma_class = sigma[k,:,:]
        scores[:,k] = scipy.stats.multivariate_normal.logpdf(X,mu_class,sigma_class)
    scores += log_prior
    max_index = np.argmax(scores, axis=1)
    prediction = np.array([classes[i] for i in max_index])
    return prediction
    

def LDA2DGaussGM(filename,num_crossval):
    dataframe = pd.read_csv(filename)
    data = np.array(pd.DataFrame(dataframe))
    np.random.shuffle(data)
    v = projection_vector(data)
    X = data[:,:-1]
    Y = data[:,-1]
    X = np.dot(X,v.real)
    data = np.column_stack((X,Y))
    
    k = num_crossval
    errors_train=np.zeros((k,))
    errors_test=np.zeros((k,))
    
    fold_size = int(data.shape[0]/k)


    for i in range(k):
        if i<k-1:
            data_test=data[i*fold_size:(i+1)*fold_size,:]
            data_train=np.vstack((data[:i*fold_size,:],data[(i+1)*fold_size:,:]))
            
        else:
            data_test=data[i*fold_size:,:]
            data_train=data[:i*fold_size,:]
            
        y_test = data_test[:,-1]
        y_train = data_train[:,-1]
        
        mu_train = get_mu(data_train)
        sigma_train = get_sigma(data_train)
        prior_train = get_prior(data_train[:,-1])
        
        train_y_ = predict_class(data_train,mu_train,sigma_train,prior_train)
        test_y_ = predict_class(data_test,mu_train,sigma_train,prior_train)
        
        error_train= y_train[y_train!=train_y_].shape[0]/float(data_train.shape[0])
        error_test= y_test[y_test!=test_y_].shape[0]/float(data_test.shape[0])
        
        
        errors_train[i] = error_train*100
        errors_test[i] = error_test*100
    
    for i in range(k):
        print("Train error for cross_validation_iteration", i+1, "is ",errors_train[i])
        print("Test error for fold", i+1, "is ",errors_test[i],"\n")
    print("Standard Deviation for Test errors is :" ,np.std(errors_test))
    
    
    
import pandas as pd
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt


def IRLS(y, X, cls, classes):   
    threshold = 0.0001
    iterations = 200
    epsilon = 0.05
    n = X.shape[0]

    threshold_array = np.ones((1,n))*epsilon
    r = np.repeat(1,n)
    R = np.diag(r)
    Wi = np.dot(np.linalg.pinv(np.dot(np.dot(X.T,R),X)),(np.dot(np.dot(X.T,R),y)))
    
    for it in range(iterations):
        prev_W = Wi
        delr =  abs(y - X.dot(Wi)).T
        r = 1.0/np.maximum( threshold_array, delr )
        R = np.diag(r[0])
        Wi = np.dot(np.linalg.pinv(np.dot(np.dot(X.T,R),X)),(np.dot(np.dot(X.T,R),y)))
        t = sum( abs(Wi - prev_W) ) 
        if t<threshold:
            return Wi
        
        
def train(data,classes):
    y = data[:,-1]
    W = np.random.normal(0,0.001,(len(classes),data.shape[1]-1))
    for i in classes:
        i = int(i)
        one_y_vs_all = np.array([1 if c==i else 0 for c in y])
        W[i,:] = IRLS(one_y_vs_all,data[:,:-1], i, 1000)
    return W
 
          
def predict_class(data,W,classes):
    X = data[:,:-1]
    n_c = len(classes)
    
    scores=np.zeros((X.shape[0],n_c))
    scores = np.dot(X,W.T)
    for i in range(scores.shape[0]):
        x = scores[i,:]
        softmax = np.exp(x) / np.sum(np.exp(x), axis=0)
        scores[i,:] = softmax
    
    max_index=np.argmax(scores,axis=1)   
    prediction = [classes[idx] for idx in max_index]
    return prediction

    


def logisticRegression(filename,num_splits,train_percent):
    dataframe = pd.read_csv(filename)
    data = np.array(pd.DataFrame(dataframe))
#    data = np.concatenate((np.ones((data.shape[0], 1)), data))      #Adding Bias
    data[:,:-1] = data[:,:-1]+np.random.normal(0, 0.001, data[:,:-1].shape)
    target = data[:,-1]
    classes = np.unique(target)
    if len(classes) > 10 :
         median = np.median(target)
         target_new = [0 if i<=median else 1 for i in target]
         data[:,-1] = target_new
         classes = np.unique(data[:,-1])
         
    data_classes = []
    
    error_matrix_test = np.zeros((num_splits,len(train_percent)))
    for k in classes:
        k = int(k)
        data_classes.append(data[target==k,:])
    
    for n in range(num_splits):
        data_train = np.zeros((1,data.shape[1]))
        data_test = np.zeros((1,data.shape[1]))
        np.random.seed(n+1)
        
        for k in classes:
            k = int(k)
            np.random.shuffle(data_classes[k])
            temp = np.vstack((data_train,data_classes[k][:int(data_classes[k].shape[0]*(.80)),:]))
            data_train = temp
            temp = np.vstack((data_test,data_classes[k][(int(data_classes[k].shape[0]*(.80)))+1:,:]))
            data_test = temp
        data_train = data_train[1:,:]
        data_test = data_test[1:,:]
        np.random.shuffle(data_train)
        
        
        for idx in range(len(train_percent)):
            t = train_percent[idx]
            data_train_t = data_train[:int(data_train.shape[0]*(t/100)),:]
            W = train(data_train_t,classes)
#            train_y_ = predict_class(data_train_t,mu_train,sigma_train,prior_train,classes)
            test_y_ = predict_class(data_test,W,classes)
            
#            y_train = data_train_t[:,-1]
            y_test = data_test[:,-1]
            
            
#           error_train= y_train[y_train!=train_y_].shape[0]/float(data_train_t.shape[0])
            error_test= y_test[y_test!=test_y_].shape[0]/float(data_test.shape[0])
            
            error_matrix_test[n,idx] = error_test*100
            print("Split: ",n+1," Train_Percentage: ",t," Test_Error: ",error_matrix_test[n,idx])
        print()
            
    for idx in range(len(train_percent)):
        print("Test_Error for Train Percentage ",train_percent[idx]," across all splits: ",np.mean(error_matrix_test[:,idx]))
            
    
    
    x = train_percent
    y = [np.mean(error_matrix_test[:,idx]) for idx in range(len(train_percent))]
    y_std = [np.std(error_matrix_test[:,idx]) for idx in range(len(train_percent))]
    plt.figure()
    plt.errorbar(x, y,yerr=y_std)
    plt.ylabel('Test_Error_Percentage')
    plt.xlabel('Training_Percentage')
    plt.title('Logitic_Regression_HomeVal50')
    plt.show()
    

    
    
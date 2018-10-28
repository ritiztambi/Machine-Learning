import pandas as pd
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt

def get_mu(data,classes):
    X = data[:,:-1]
    Y = data[:,-1]
    n_c = len(classes)
    f = X.shape[1]
    mu = np.zeros((n_c,f))
    for k in classes:
        k = int(k)
        X_class = X[Y==k, :]
        mu[k,:] = np.mean(X_class, axis=0)
    return mu
    
   
def get_sigma(data,classes):
    X = data[:,:-1]
    Y = data[:,-1]
    n_c = len(classes)
    f = X.shape[1]
    sigma = np.zeros((n_c,f))
    for k in classes:
        k = int(k)
        X_class = X[Y==k, :]
        for d in range(f):
            sigma[k,d] = np.var(X_class[:,d])    
    sigma[sigma==0] = 1e-03
    return sigma
    

def get_prior(data,classes):
    Y = data[:,-1]
    len_data = Y.shape[0]
    prior = np.array([np.sum(Y == cls) / len_data for cls in classes])
    return prior


def predict_class(data,mu,sigma,prior,classes):
    X = data[:,:-1]
    log_prior = np.array([math.log(p) for p in prior])
    n_c = len(classes)
    scores = np.zeros((data.shape[0],n_c))
    for k in classes:
        k = int(k)
        for j in range(X.shape[1]):
            scores[:,k] += scipy.stats.norm.logpdf(X[:,j],loc=mu[k, j], scale=math.sqrt(sigma[k, j]))
    scores += log_prior

    max_index = np.argmax(scores, axis=1)
    prediction = np.array([classes[i] for i in max_index])
    return prediction
    

def naiveBayesGaussian(filename,num_splits,train_percent):
    dataframe = pd.read_csv(filename)
    data = np.array(pd.DataFrame(dataframe))
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
            mu_train = get_mu(data_train_t,classes)
            sigma_train = get_sigma(data_train_t,classes)
            prior_train = get_prior(data_train_t,classes)
            
#            train_y_ = predict_class(data_train_t,mu_train,sigma_train,prior_train,classes)
            test_y_ = predict_class(data_test,mu_train,sigma_train,prior_train,classes)
            
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
    plt.title('NaiveBayes')
    plt.show()
    

    
import pandas as pd
import numpy as np
from cvxopt import matrix, solvers
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



def train(X,Y,c):
    X = X/255
    solvers.options['show_progress'] = False
    ''' Get Lagrangians '''
    threshold = 1e-3
  
    n_samples,n_features = X.shape
    
    Y = Y.reshape(-1,1)
    K = np.multiply(Y,X)
    K = np.dot(K,K.T)
    P = matrix(K,(K.shape[0],K.shape[1]),'d')
    Q = matrix(-1*np.ones((n_samples,1)))
    
    tmp1 = -1*np.eye(n_samples)
    tmp2 = np.eye(n_samples)
    G = matrix(np.vstack((tmp1, tmp2)))
    
    tmp1 = np.zeros(n_samples)
    tmp2 = np.ones(n_samples) * c
    H = matrix(np.hstack((tmp1, tmp2)))
    
    A = matrix(Y.reshape(1,-1),(1,n_samples),'d')
    B = matrix(0.0)
   
    
    
    solution = solvers.qp(P, Q, G, H, A, B)
    alpha = np.array(solution['x'])
    
   
    indx = np.where(alpha>threshold)
    
    alpha[alpha < threshold] = 0
    sv = alpha[alpha!=0]
    sv_len = sv.shape[0]
    
    w = np.dot(np.multiply(alpha,Y).T,X)
    w = w.reshape(1,-1)
   
    
    b = Y[indx[0]] - np.dot(X[indx[0]], w.T)
    b = np.mean(b, axis=0)
    
    return w,b,sv_len


def predict(X,w,b):
    result =[]
    for x in X:
        res = np.sign(np.dot(w,x)+b)
        result.append(res)
    return result

def prediction_error(pred,target):
    pred = np.array(pred).reshape(-1,1)
    target.reshape(-1,1)
    check_array = np.array([y_ == y for (y_,y) in zip(pred,target)])
    mistakes = target.shape[0] - sum(check_array)
    error = (1-(np.sum(check_array))/len(target))*100
    return error,mistakes
#
#def calculate_mistakes(pred,target):

def MyDualSVM(filename,C):
    
    dataframe = pd.read_csv(filename);
    data = np.array(pd.DataFrame(dataframe))
    target = data[:,0]
    ''' Relabel Y as +1 and -1 '''
    classes = np.unique(target)
    label_1 = classes[0]
    label_2 = classes[1]
    data[target==label_1,0] = 1;
    data[target==label_2,0] = -1;
    
    
    
    num_runs = 10
    test_error_matrix = np.zeros((num_runs,len(C)))
    test_mistakes_matrix = np.zeros((num_runs,len(C)))
    margin_matrix = np.zeros((num_runs,len(C)))
    train_error_matrix = np.zeros((num_runs,len(C)))
    train_mistakes_matrix = np.zeros((num_runs,len(C)))
    support_vectors_matrix = np.zeros((num_runs,len(C)))
    
    c_index = 0
    for c in C:
         print("-------Running for C:",c,"-------")
         print()
         print()
         X = data[:,1:]
         y = data[:,0]
#         fold_len  = int(data.shape[0]/10)
#         folds = []
#         for i in range(10):
#            folds.append(data[i*fold_len:(i+1)*fold_len,:])
         for run in range(num_runs):
             print("Run_Iteration:",run+1)
             X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
                                
             w,b,sv_len = train(X_train,Y_train,c)
             margin_matrix[run,c_index] = 1/np.linalg.norm(w)
             support_vectors_matrix[run,c_index] = sv_len
             y_train_pred = predict(X_train,w,b)
             train_error,train_mistakes = prediction_error(y_train_pred,Y_train)
             train_error_matrix[run,c_index] = train_error
             train_mistakes_matrix[run,c_index] = train_mistakes
             
             
             y_test_pred = predict(X_test,w,b)
             test_error,test_mistakes = prediction_error(y_test_pred,Y_test)
             test_error_matrix[run,c_index] = test_error
             test_mistakes_matrix[run,c_index] = test_mistakes
             
             print("Test Error:",test_error)
             print("Test Mistakes:", test_mistakes)
             
             print()
         c_index+=1
     
    print("Summary and Plots")
    print("---------------")
    print("Average Test_Errors:")
    mean = np.mean(test_error_matrix,axis=0)
    std = np.std(test_error_matrix,axis=0)
    for c,t,z in zip(C,mean,std):
        print("C: {0} Test_error: {1} Standard_Deviation: {2}".format(c,t,z))
    plt.plot(C,mean,'o-')
    plt.xlabel("C")
    plt.ylabel("Error")
    plt.title("Test_Errors")
#    plt.savefig("Test_Errors")
    plt.show()
        
    
        
    print("---------------")
    print("Average Test_Mistakes:")
    mean = np.mean(test_mistakes_matrix,axis=0)
    for c,t in zip(C,mean):
        print("C: {0} Test_Mistakes: {1}".format(c,t))
    plt.plot(C,mean,'o-')
    plt.xlabel("C")
    plt.ylabel("Mistakes")
    plt.title("Test_Mistakes")
#    plt.savefig("Test_Mistakes")
    plt.show()  
        
   
    
    print("---------------")
    print("Average Margins:")
    mean = np.mean(margin_matrix,axis=0)
    for c,t in zip(C,mean):
        print("C: {0} Average_Margin: {1}".format(c,t))
    plt.plot(C,mean,'o-')
    plt.xlabel("C")
    plt.ylabel("Margins")
    plt.title("Average_Margins")
#    plt.savefig("Average_Margins")
    plt.show()   
        
    print("---------------")
    print("Average Number of Support Vectors:")
    mean = np.mean(support_vectors_matrix,axis=0)
    for c,t in zip(C,mean):
        print("C: {0} Average_Num_SupportVectors: {1}".format(c,t))
    plt.plot(C,mean,'o-')
    plt.xlabel("C")
    plt.ylabel("Num_SV")
    plt.title("Number_of_SupportVectors")
#    plt.savefig("Number_of_SupportVectors")
    plt.show()     
        
#if __name__ == '__main__':
#    C = [0.01,0.1,1,10,100]
#    MyDualSVM('MNIST-13.csv',C)
#    
    
        
    
# GaussianGenerativeModel.py

import numpy as np
from scipy.stats import multivariate_normal as mvn  


# Implement a Gaussian Generative Model from scratch


class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False, mu = None, cov = None, pi = None, k=3):
        self.is_shared_covariance = is_shared_covariance
        self.mu = mu
        self.cov = cov
        self.pi = pi 
    
    # mle for dataset pi_k
    def __find_prior_pi(self, y):
        y = list(map(np.argmax, y))
        unique, counts = np.unique(y, return_counts=True)
        t = counts / len(y)
        self.pi = t
        return t
    
    # mle for data set mu_k
    def __find_mu(self, X,y):
        y = list(map(np.argmax, y))
        # finds class indices
        cl_0 = [i for i, num in enumerate(y) if num == 0]
        cl_1 = [i for i, num in enumerate(y) if num == 1]
        cl_2 = [i for i, num in enumerate(y) if num == 2]
        
        X_0 = []
        X_1 = []
        X_2 = []

        for i in range(X.shape[1]):
            X_0.append(np.mean(X[cl_0, i])) 
            X_1.append(np.mean(X[cl_1, i]))
            X_2.append(np.mean(X[cl_2, i]))
        
        mu = [X_0, X_1, X_2]
        self.mu = mu
            
    def __find_sigma(self, X, ys, mu):
        y = list(map(np.argmax, ys))
        cl_0 = [i for i, num in enumerate(y) if num == 0]
        cl_1 = [i for i, num in enumerate(y) if num == 1]
        cl_2 = [i for i, num in enumerate(y) if num == 2]
        
        cov0 = np.cov(X[cl_0].T, bias = True)
        cov1 = np.cov(X[cl_1].T, bias = True)
        cov2 = np.cov(X[cl_2].T, bias = True)
    
        if self.is_shared_covariance:
            cv = np.cov(X.T, bias = True)
            cov = [cv, cv, cv]
        else:
            cov = [cov0, cov1, cov2]
        
        self.cov = cov
        
        return cov
        
    
    def __onehot(self, y):
        n_values = np.max(y) + 1
        y_enc = np.eye(n_values)[y]
        return y_enc
                

    # fit function finds mle for gaussian parameters
    def fit(self, X, y):
        y = self.__onehot(y)
        self.__find_mu(X, y)
        self.__find_sigma(X, y, self.mu)
        self.__find_prior_pi(y)
        

    # predict function
    def predict(self, X_pred):
        preds = []
        cl1 = []
        cl2 = []
        cl3 = []
        
        for i in range(X_pred.shape[0]):
            cl1.append(mvn.pdf(X_pred[i], self.mu[0], self.cov[0]))
            cl2.append(mvn.pdf(X_pred[i], self.mu[1], self.cov[1]))
            cl3.append(mvn.pdf(X_pred[i], self.mu[2], self.cov[2]))
                       
        y_mat = np.vstack((cl1,cl2,cl3)).T
        y_p = np.array(list(map(np.argmax, y_mat)))
        
        return y_p

    # implementation of negative log likelihood
    def negative_log_likelihood(self, X, y):
        y = self.__onehot(y)
        X = np.array(X)
        likel = []
        i = 0
        for elt in list(map(np.argmax, y)):
            if np.argmax(elt) == 0:
                likel.append(np.log(self.pi[0]*mvn.pdf(X[i], self.mu[0], self.cov[0])))
            elif np.argmax(elt) == 1:
                likel.append(np.log(self.pi[1]*mvn.pdf(X[i], self.mu[1], self.cov[1])))
            else:
                likel.append(np.log(self.pi[2]*mvn.pdf(X[i], self.mu[2], self.cov[2])))
            i += 1
                          
        return -np.sum(likel)
                
        

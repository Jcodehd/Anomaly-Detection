from statistics import mean
import numpy as np



def estimateGaussian(X):

    m = len(X);  
    X = np.mat(X);
    mu = np.mean(X, axis=0); #平均值
    
    sigma = np.mean(np.power(X-np.repeat(mu, m, axis=0), 2), axis=0);

    return mu, sigma;


    
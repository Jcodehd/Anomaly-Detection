from cmath import exp, pi
import numpy as np;
from Diag import diag;
import cv2

def multivariateGaussian(X, mu, sigma):
    m = X.shape[0];
    n = X.shape[1];
    X = np.mat(X);

    if sigma.shape[0] == 1 or sigma.shape[1] == 1:
        sigma = diag(sigma);


    sigma_det = np.linalg.det(sigma);
    sigma_inv = np.linalg.inv(sigma);
    X = X - np.repeat(mu, m, axis=0);


    p = pow(2*pi, -n/2)*pow(sigma_det, -0.5)*np.exp(-0.5*np.sum(cv2.multiply(np.dot(X, sigma_inv),X), axis=1));
    

    return p;

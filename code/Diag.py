import numpy as np;


def diag(X):

    m = X.shape[1];

    diag_ = np.mat(np.zeros((m,m)));


    for i in range(m):

        diag_[i,i] = X[0,i];

    return diag_;
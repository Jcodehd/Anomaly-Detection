import numpy as np;


def selectThreshold(yval, pval):

    bestF1 = 0;
    bestE = 0;
    pval_max = max(pval);
    pval_min = min(pval);
    m = len(yval);

    tp = 0;
    fp = 0;
    fn = 0;

    for e in np.linspace(pval_min, pval_max, num=10000):
        
        predict = (pval < e);
        
        for i in range(m):
            
            if predict[i] == 1 and yval[i,0] == 1:
                tp = tp + 1;

            if predict[i] == 1 and yval[i,0] == 0:
                fp = fp + 1;
            
            if predict[i] == 0 and yval[i,0] == 1:
                fn = fn + 1

        if tp == 0:
            continue;

        p = tp/(tp+fp);
        r = tp/(tp+fn);
        f1 = 2*p*r/(p+r);

        if f1 >= bestF1:
            bestF1 = f1;
            bestE = e;

    
    return bestE, bestF1;
        



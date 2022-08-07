from cmath import pi
import scipy.io as sc
import numpy as np;
from estimateGaussian import estimateGaussian;
from multivariateGaussian import multivariateGaussian;
from selectThreshold import selectThreshold;

# 加载数据
data = sc.loadmat('Machine Learning/Anomaly Detection/ex8data2.mat');
X = data['X'];
Xval = data['Xval'];
yval = data['yval'];


# 求mu、sigma
mu, sigma = estimateGaussian(X);

# 计算预测值
pval = multivariateGaussian(Xval, mu, sigma);
p  = multivariateGaussian(X, mu, sigma);

# 计算阈值

e, F1 = selectThreshold(yval, pval);

print("F1分数为", F1);
print("异常点数为", np.sum(p<e));








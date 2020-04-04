#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd

def mean(X):
    return X.sum()/X.shape[0]

def expected_value(X):
    return X.mean()

def variance(X,ddof=1):
    """    
    Compute the variance; dedicated to @craq21
    Calculates variance,a measure of spread of a distribution
    
    Arguments:
        X - pd.Series or pd.DataFrame
        ddof - delta degrees of freedom; 1 by default for calculating sample variance, 0 for maximum likelihood estimate of the variance for normally distributed variables
    
    Returns:
        variance
    """
    mu = mean(X)
    diffx = X-mu
    return  (np.sum(diffx**2))/(X.shape[0]-ddof)

def stddev(X):
    return variance(X)**0.5

def cov(x,y):
    res = (x - mean(x))*(y-mean(y))
    return res.sum()/(x.shape[0]-1)


# In[ ]:





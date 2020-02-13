#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from scipy import stats


# In[2]:


def mean(X):
    return X.sum()/X.shape[0]

def expected_value(X):
    return X.mean()

def variance(X):
    mu = mean(X)
    diffx = X-mu
    return  (np.sum(diffx**2))/(X.shape[0]-1)

def stddev(X):
    return variance(X)**0.5

def mode(X):
    modes = []
    for col in X.columns:
        modes.append(pd.value_counts(X[col]).keys()[0])
    return pd.Series(modes,X.columns)
    
def median(X):
    l = X.shape[0]
    medians = []
    for col in X.columns:
        sorted_vals = X[col].sort_values()
        if l%2==0:
            medians.append((sorted_vals[l//2-1]+sorted_vals[l//2])*0.5)
        else:
            medians.append(sorted_vals[l//2])
    
    return pd.Series(medians,X.columns)
    
def skewness(X):
    skews = []
    means = X.mean()

    for col in X.columns:
        diffx,n = X[col]-X[col].mean(), X.size
        
#        coeff = np.sqrt(n*(n-1.0))/(n-2.0) 
#        moments = (np.sum(diffx**3)) / (np.sum(diffx**2) ** 1.5)
#        skew = coeff * moments
        
        skew = (1/n * np.sum(diffx**3))/((1/(n-1) * np.sum(diffx**2))**1.5)
        skews.append(skew)
        
    return pd.Series(skews,X.columns)

def kurtosis(X):
    kurts = np.array([])
    mu,var = mean(X),variance(X)
    for col in X.columns:
        
        diffx,n = X[col]-mu[col], X[col].size
        kurt = (1/n * np.sum(diffx**4)) / (1/n * np.sum(diffx**2))**2
        kurts = np.append(kurts,kurt-3)
        
    return pd.Series(kurts,X.columns)


# In[ ]:


def sample_size_avg_score(s,delta,t=1.96):
    """
    Calculates minimum sample size for a certain margin of error for an avg score test
    
    Arguments:
        delta -- margin of error
        s -- stddev
        t=1.96 -- z-score for 0.95 confidence interval
    
    Returns:
        n -- sample size
    """
    return np.square(t*s/delta)

def margin_of_error_avg_score(n,s,t=1.96):
    """
    Calculates margin of error for a certain minimal sample size in an avg score test
    
    Arguments:
        n -- sample size
        s -- stddev
        t=1.96 -- z-score for 0.95 confidence interval
        
    Returns:
        delta -- margin of error
    """
    return t*s/np.sqrt(n)

def sample_size_proportion(delta,p,t=1.96):
    """
    Calculates minimum sample size for a certain margin of error for a proportion test
    
    Arguments:
        delta -- margin of error, 0<delta<1
        p -- proportion (chance to succeed | sample class share | etc)
        t=1.96 -- z-score for 0.95 confidence interval
    
    Returns:
        n -- sample size
    """
    return np.square((t*np.sqrt(p*(1-p)))/delta)

def margin_of_error_proportion(n,p,t=1.96):
    """
    Calculates margin of error 
    
    Arguments:
        n -- sample size
        p -- proportion (chance to succeed | sample class share | etc)
        t=1.96 -- z-score for 0.95 confidence interval
        
    Returns:
        delta -- margin of error
    """    
    return t * np.sqrt(p*(1-p)/n)


# In[3]:


# x = 50 *np.random.rand(20,1)
# y = 100*np.random.rand(20,1)
# z = 200*np.random.rand(20,1)
# t = pd.DataFrame(data=np.concatenate([x,y,z],axis=1), columns=['a','b','c'])
# v1 = skewness(t)
# v2 = stats.skew(t)
# v1,v2
# v2/v1.values


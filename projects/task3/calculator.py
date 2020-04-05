#!/usr/bin/env python
# coding: utf-8

# In[45]:


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

class RankMethods:
    """
    A set of static methods to calculate sample ranks
    
    Example:
        input @ x=[3,5,6,11], y=[3,1,2,1]
        output@ sum of squared differences
        code:
            rm = RankMethods()
            r1,r2 = rm.rank(np.array([])),rm.rank(np.array([]))
            diff = rm.rank_difference(r1,r2)
            sum_of_squared_diff = rank_difference_sum(r1,r2)

    """
    @staticmethod
    def rank(x,return_dict=False):
        """
        Ranks an array [items]
        Calculates: 
                a dictionary {item:rank}

        Returns:
                a list of ranks
        Example:
            input@[3,1,2,1]
            output@{3:1,1:3.5,2:2}
        """
        ranks_dict = {el:[] for el in x}
        sorted_x = sorted(x,reverse=True)

        for i in range(len(sorted_x)):
            ranks_dict[sorted_x[i]].append(i+1)

        for k,v in ranks_dict.items():
            if len(v)>1:
                ranks_dict[k] = mean(np.array(v))
            else:
                ranks_dict[k] = v[0]

        ranks = np.array([ranks_dict[item] for item in x])

        if return_dict:
            return ranks, ranks_dict
        return ranks
    
    @staticmethod
    def rank_difference(rank_x,rank_y):
        assert rank_x.shape[0]==rank_y.shape[0]
        
        return np.array([r1 - r2 for r1,r2 in zip(rank_x,rank_y)])
    
    @staticmethod
    def rank_difference_sum(rank_x,rank_y):
        assert rank_x.shape[0]==rank_y.shape[0]

        diff = rank_difference(rank_x,rank_y)
        return np.square(diff).sum()
    
    @staticmethod
    def areDistinct(x,y):
        assert x.shape[0]==y.shape[0]
        
        for i in range(x.shape[0]):
            if x[i]==y[i]:
                return False
        return True    

def concordant_discordant(x):
    concordant,discordant =[],[]
    
    for i in range(len(x)-1):
        tmp = x[i+1:]
        c = sum(1 for el in tmp if el>x[i])
        d = tmp.shape[0]-c
        
        concordant.append(c)
        discordant.append(d)
    
    return np.array(concordant), np.array(discordant)

def concordant_discordant_ties(x,y):
    c,d,t_x,t_y,t_b = 0,0,0,0,0
    
    n = x.shape[0]
    
    for i in range(n-1):
        for j in range(1,n):
            if (x[i]>x[j] and y[i]>y[j]) or (x[i]<x[j] and y[i]<y[j]):
                c+=1
            elif (x[i]<x[j] and y[i]>y[j]) or (x[i]>x[j] and y[i]<y[j]):
                d+=1
            if (x[i]==x[j] or y[i]==y[j]):
                if x[i]==x[j]:
                    t_x+=1
                if y[i]==y[j]:
                    t_y+=1
                if x[i]==x[j] and y[i]==y[j]:
                    t_b+=1
    return c,d,t_x-t_b,t_y-t_b


# In[50]:


#Example
r = np.array([1,2,4,3,6,5,8,7,10,9,12,11])
c,d= concordant_discordant(r)


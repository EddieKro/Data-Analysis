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


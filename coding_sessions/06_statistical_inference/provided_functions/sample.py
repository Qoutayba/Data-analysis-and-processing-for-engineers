import numpy as np


def flip_coin( n=1, datatype='string', weight=0.6):
    """
    flip a possibly non-fair coin and return the results
    can return a list of strings or an array of 0 and 1
    Parameters:
    -----------
    n:              int
                    how long the return list is
    datatype:       string, default 'string'
                    return type of the results, 
                    accepts string -> list
                    or numbers/array -> array 
    weight:         float, default 0.6 
                    'fairness' of the coin, denoted theta in the lecture
                    0.5 is a fair coin, value should be in interval [0, 1]
    Returns:
    --------
    flip            list or numpy 1d-array
                    result of the coin flip 
    """
    result = np.random.rand( n)
    flip = result < weight
    if datatype.lower() in ['array', 'numbers' ]:
        return flip.astype( int)
    elif datatype == 'string':
        flip = list( flip)
        for i in range( n):
            return [ 'tails' if x else 'heads' for x in flip ]



def uniform_distribution( n=1, a=0.5, b=2):
    """
    sample n values out of the uniform distribution
    shadows the numpy function
    Parameters:
    -----------
    a:          float, default 0
                lower bound of the distribution
    b:          float, default 1
                upper bound of the distribution
                expected value of the normal distribution
    interval:   list of 2 floats, default None
                interval on which to plot the distribution 
    Returns:
    --------
    x:          numpy 1d-array
                samples of the uniform distribution
    """
    x = np.random.uniform( a, b, n) 
    return x

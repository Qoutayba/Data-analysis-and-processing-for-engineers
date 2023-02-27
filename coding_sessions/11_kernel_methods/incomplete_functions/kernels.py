import numpy as np

from numpy import exp
from numpy.linalg import norm
from math import ceil, floor

def gaussian_1d(  x, x_prime, gamma):
    """
    Computes the gaussian kernel function of 'x' and 'x_prime'
    uses the equation " k(x, x_prime) =  exp( -gamma * || x- x_prime||^2 )
    Only usable if x and x_prime are 1d-vectors of multiple samples
    Parameters:
    -----------
    x:          nd numpy array
                vector of a training data/support points 
                (if nd-array -> each column one sample)
    x_prime:    nd numpy array or float
                vector of a validation sample
                (if nd-array -> each column one sample)
    gamma:      float
                non-negaitve adjustment parameter
    Returns:
    --------
    k(x, x_prime)   float or nd-array
                    evaluation of the kernel function 
    """
    if x.ndim > 1 or x_prime.ndim > 1:
        raise Exception( 'error in "kernels.gaussian_1d",'+
        'single samples must be scalar valued, use the kernels.gaussian function instead' )

    n_1 = len(x)
    #n_2 = #TODO
    #kernel_matrix = np.zeros( ( #TODO ))
    #for i in range( n_1):
        #TODO
    return #TODO


def gaussian( x, x_prime, gamma ):
    """
    Computes the gaussian kernel function of 'x' and 'x_prime'
    uses the equation " k(x, x_prime) =  exp( -gamma * || x- x_prime||^2 )
    This function differentiates between different dimensional inputs,
    it is assumed that 1d-vectors always define 1 sample
    Parameters:
    -----------
    x:          nd numpy array
                vector of a training data/support points 
                (if nd-array -> each column one sample)
    x_prime:    nd numpy array or float
                vector of a validation sample
                (if nd-array -> each column one sample)
    gamma:      float
                non-negaitve adjustment parameter
    Returns:
    --------
    k(x, x_prime)   float or nd-array
                    evaluation of the kernel function 
                    if x and x_prime are nd-arrays, returns the a evaluation matrix
    """
    if isinstance( x_prime, float) or isinstance( x_prime, int ):
        x_prime = np.array( [x_prime] ) 

    ## if conditions: differentiate between number of samples
    # only one sample
    if x.ndim == 1 and x_prime.ndim == 1: #only possible if samples are 1d vector
        print( 'x and x_prime were assumed to be 1 sample of n-d vectors')
        print( 'if x and x_prime are 1d-vectors of multiple samples, use gaussian_1d instead!' ) 
        return exp( -gamma * norm( x - x_prime)**2) 
    #multiple samples
    elif x.ndim == 2:
        if x_prime.ndim == 1: #only one sample in x_prime
            norm_sample = norm( x_prime, axis=0)**2
        elif x_prime.ndim >=1: #multiple samples
            norm_sample = norm( x_prime, axis=1)**2 
        norm_support = norm( x, axis=1)**2
        norm_matrix = ((norm_sample - 2*x @ x_prime.T).T  + norm_support).T 
        # efficient implementation for the computation of the term in the norm
        return exp( -gamma * norm_matrix ) 
    else:
        raise Exception( 'error in "kernels.gaussian",'+
        'x must have more than 1 sample if x_prime has multiple')



def linear( x, x_prime, slope ):
    """
    Define the linear kernel as max( 1- | x - x'|*slope, 0)
    Parameters:
    -----------
    x:          nd numpy array
                vector of a training data/support points 
                (if nd-array -> each column one sample)
    x_prime:    nd numpy array or float
                vector of a validation sample
                (if nd-array -> each column one sample)
    slope:      float
                slope of the kernel function
    Returns:
    --------
    k(x, x_prime)   float or nd-array
                    evaluation of the kernel function 
                    if x and x_prime are nd-arrays, returns the a evaluation matrix 
    """
    if (x.ndim == 2 and x_prime.ndim == 2) or (x.ndim == 1 and x_prime.ndim == 1):
        kernel_matrix = np.zeros( ( x.shape[0], x_prime.shape[0] ) )
        for i in range( x.shape[0]):
            for j in range( x_prime.shape[0]):
                kernel_matrix[i,j] = np.maximum( 1- (norm(x[i]-x_prime[j])*slope), 0)
    elif x.ndim == 2 and x_prime.ndim == 1:
        kernel_matrix = np.zeros(  x.shape[0]  )
        for i in range( x.shape[0]):
            kernel_matrix[i] = np.maximum( 1- (norm(x[i]-x_prime)*slope), 0) 
    else:
        x, x_prime = x_prime, x
        kernel_matrix = np.zeros(  x.shape[0]  )
        for i in range( x.shape[0]):
            kernel_matrix[i] = np.maximum( 1- (norm(x[i]-x_prime)*slope), 0) 
    return kernel_matrix

    
def polynomial( x, x_prime, slope, constant=1, poly_order=2 ):
    """
    Compute the polynomial kernel defined as 
    (slope * max(1-|x-x_prime|,0) +constant) ** poly_order
    Parameters:
    -----------
    x:          nd numpy array
                vector of a training data/support points 
                (if nd-array -> each column one sample)
    x_prime:    nd numpy array or float
                vector of a validation sample
                (if nd-array -> each column one sample)
    slope:      float
                slope of the kernel function
    constant:   float
                added constant in the kernel function
    poly_order: int
                polynomial order of the kernel function
    Returns:
    --------
    k(x, x_prime)   float or nd-array
                    evaluation of the kernel function 
                    if x and x_prime are nd-arrays, returns the a evaluation matrix 
    """
    if (x.ndim == 2 and x_prime.ndim == 2) or (x.ndim == 1 and x_prime.ndim == 1):
        kernel_matrix = np.zeros( ( x.shape[0], x_prime.shape[0] ) )
        for i in range( x.shape[0]):
            for j in range( x_prime.shape[0]):
                kernel_matrix[i,j] = (slope*np.max( (1-norm(x[i]-x_prime[j]), 0)) + constant )**poly_order
    elif x.ndim == 2 and x_prime.ndim == 1:
        kernel_matrix = np.zeros(  x.shape[0]  )
        for i in range( x.shape[0]):
            kernel_matrix[i] = (slope*np.max( (1-norm(x[i]-x_prime), 0)) + constant )**poly_order
    else:
        raise Exception( 'unexpected input in kernels.polynomial, x has have multiple samples if x_prime has multiple')
    return kernel_matrix

    

def wendland( x, x_prime, gamma, k=1 ):
    if k!= 1 or k!= 0:
        raise Exception( 'k only allows for values {0, 1}')

    if isinstance( x_prime, float):
        x_prime = np.array( x_prime)
    if x.ndim == 1:
        error_message = 'Function does not work for 1d-vectors of multiple scalar valued samples, '
        error_message = error_message + 'pass x[:,None] & x_prime[:,None] to thin function for '
        error_message = error_message + 'it to work on scalar valued problems'
        raise Exception( error_message)

    d = x.shape[1]
    l = floor( d/2 ) +k+1 
    if (x.ndim == 2 and x_prime.ndim == 2) or (x.ndim == 1 and x_prime.ndim == 1):
        kernel_matrix = np.zeros( (x.shape[0], x_prime.shape[0] ) )
        for i in range( x.shape[0]):
            for j in range( x_prime.shape[0] ):
                activation = norm( x[i,:] - x_prime[j,:] ) / gamma
                p_n_1 =  (l+1) * activation + 1
                kernel_matrix[i,j] = np.max( [0,(1-activation)] )**(l + k) * p_n_1
    elif x.ndim ==2 and x_prime.ndim == 1:
        kernel_matrix = np.zeros( x.shape )
        for i in range( x.shape[0]):
            activation = norm( x[i,:] - x_prime ) / gamma
            p_n_1 =  (l+1) * activation +1
            kernel_matrix[i] = np.max( [0, (1-activation) ])**(l + k) * p_n_1
    else:
        raise Exception( 'unexpected input in kernels.wendland, x has have multiple samples if x_prime has multiple')
    return kernel_matrix


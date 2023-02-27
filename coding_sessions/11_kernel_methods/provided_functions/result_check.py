import numpy as np
import sample_sets as sample
from numpy import exp
from numpy.linalg import norm
from kernels import gaussian

def Task_1a( x_negative, x, y_hat):
    """ check if the data has been correctly split by their sign """
    if len( x[y_hat<=0]) != len( x_negative):
        raise Exception ('Data wrongly split, did not get the right amount of samples')
    if not (x_negative==x[y_hat<0]).all():
        raise Exception ('Data wrongly split, found positive values of y_hat corresponding to x where only negatives were expected')


def data_split( x_positive, y_positive, x_negative, y_negative ):
    if (y_positive == -1 ).any():
        raise Exception( 'negative values found in y_positive, they should all be negative')
    if (y_negative == 1 ).any():
        raise Exception( 'positive values found in y_negative, they should all be positive')


def kernel_implementation( kernel_function):
    gamma = 1
    x = np.random.rand(4) *2
    x_2 = np.random.rand(5) *3
    expected_shape = (len( x), len(x_2) )
    kernel_matrix = kernel_function( x, x_2, gamma)
    fname = kernel_function.__name__
    if not np.allclose( kernel_matrix.shape, expected_shape):
        raise Exception( 'error in {}, unexpected shape, got {}, expected {}'.format( fname, kernel_matrix.shape, expected_shape) )
    sym_part = kernel_matrix[:min(expected_shape), :min(expected_shape) ]
    if fname == 'gaussian_1d' and not np.allclose( kernel_matrix[0,1], exp(-gamma* np.abs( x[0]-x_2[1])**2 ) ):
        raise Exception( 'error in {}, kernel response wrongly computed'.format( fname))
    kernel_matrix = kernel_function( x, x, gamma)
    if not np.allclose( kernel_matrix, kernel_matrix.T ):
        raise Exception( 'error in {}, expected a symmetric kernel matrix for x=x_prime, did not get one'.format( fname))



def model( evaluation_function, reference_checkerboard):
    """ evaluate the trained model and compute how many samples got wrongly classified"""
    n_samples = 500
    samples, target = sample.checkerboard( reference_checkerboard, n_samples, n_noisy=0)
    wrong = 0
    for i in range( n_samples):
        if evaluation_function( samples[i]) != target[i]:
            wrong += 1
    print( 'got {} out of {} wrong samples in the test set.'.format( wrong, n_samples) )
    print( '(if your model was perfect you would have 0 wrong samples)' )
    print( '((a good model has around {} wrong samples))'.format( n_samples//10) )



























































































































####  you are not supposed to look down here, go back to your tasks!!! ####
def gaussian_kernel( x, x_prime, gamma ):
    """
    Computes the gaussian kernel function of 'x' and 'x_prime'
    uses the equation " k(x, x_prime) =  exp( -gamma * || x- x_prime||^2 )
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
                    of shape (n_x times n_x_prime) 
    """
    ####  you are not supposed to look down here, go back to your tasks!!! ####
    if not isinstance( x_prime, np.ndarray):
        x_prime = np.array( [x_prime] ) 
    if x.ndim == 1 and x_prime.ndim == 1:
        return exp( -gamma * norm( x- x_prime)**2) 


    elif x.ndim == 2:
        if x_prime.ndim == 1: #only one evaluation sample
            norm_sample = norm( x_prime, axis=0)**2
        elif x_prime.ndim >=1: #multiple evaluation samples
            norm_sample = norm( x_prime, axis=1)**2

        norm_support = norm( x, axis=1)**2
        norm_matrix = ((norm_sample - 2*x @ x_prime.T).T  + norm_support).T
        # row and column wise addition with numpy (not so intuitive) 
        return exp( -gamma * norm_matrix ) 

    else: 
        dimension_error =""" Dimensions do not match\n
        x       - shape {}; training samples (needs to be more than 1 sample\n
        x_prime - shape {}; evaluation samples (can be either 1 or more samples)\n
        Note that the samples are arranged row wise (each row 1 sample"""
        raise Exception( dimension_error)


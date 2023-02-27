import numpy as np
from scipy import sparse
import padding as pad
import result_check as check

try:
    from scipy.special import factorial
except:
    from scipy.misc import factorial


def moving_average( signal, window_length=5):
    """
    Applies a mean filter to a given signal
    Parameters:
    -----------
    signal:         numpy 1d-array
                    noisy signal to smooth
    window_length:  int, default 5
                    total number of points to consider for smoothing
    Returns:
    --------
    signal:         numpy 1d-array
                    smoothed signal of the mean filter
    """
    if window_length % 2 == 0:
        window_length += 1
        print( 'even "window_length", defaulting to window_length + 1 -> {}'.format( window_length) )
    pad_size = window_length // 2 
    #padded_signal = #TODO #choose a padding for the signal
    #smoothed_signal = padded_signal.copy() 

    #for i in range( pad_size, len( padded_signal)-pad_size):
        #smoothed_signal[#TODO... # implement the "moving average" filter via for loop

    return #TODO #un-padded smoothed signal


def moving_average_conv( signal, window_length=5):
    """
    Applies a mean filter to a given signal
    assumes periodicity of the signal
    Parameters:
    -----------
    signal:         numpy 1d-array
                    noisy signal to smooth
    window_length:  int, default 5
                    total number of points to consider for smoothing
    Returns:
    --------
    signal:         numpy 1d-array
                    smoothed signal of the mean filter
    """
    if window_length % 2 == 0:
        window_length += 1
        print( 'even "window_length", defaulting to window_length + 1 -> {}'.format( window_length) )
    pad_size = window_length // 2 
    filter_kernel = np.zeros( signal.shape) 
    # Define window_length number of 1's at both ends of filter_kernel 
    #filter_kernel[#TODO...
    #NOTE: Normalize filter_kernel before applying convolution
    #...TODO implement the "moving average" filter via convolution

    return smoothed_signal.real 



def compute_filter( poly_order=2, window_length=None, check_flag=False ): 
    """
    Precompute the Savitzky-Golay filter for signal smoothing
    Application of the filter with 'apply_filter' 
    Parameters:
    -----------
    poly_order:     int, default 2
                    polynomial order of the filter
    window_length:  int, default None
                    number of points used for smoothing
                    defaults to window_length = 2*poly_order+1 if not specified 
    Returns
    -------
    beta:           numpy nd-array
                    shape == (poly_order+1, window_length)
                    Each row refers to the filter coefficients of the n-th derivative
                    (0th row -> no derivative, 1st row -> first derivative, etc.)
    """
    # Input check, compute default value for "window_length"
    if window_length is None:
        window_length = 2*poly_order + 1
    assert window_length > poly_order, '"poly_order" must be smaller than "window_length"'
    assert window_length % 2 != 0, '"window_length" must be uneven'
    
    P = np.zeros( (poly_order+1, window_length), dtype=int )
    xi = #TODO
    #for i in range( poly_order+1 ):
    #    P[i,:]  = #TODO ** #TODO
    #beta = #TODO #compute the filter (smoothing coefficients)

    check.savgol_filter( P, beta, poly_order, window_length, check_flag ) 
    return beta 


def apply_filter( signal, beta, derivative=[0], stepsize=None ):
    """
    Apply the precomputed Savitzky-Golay filter on the noisy signal.
    The signal is periodically padded.  
    OPTIONAL, compute the n-th derivative by specifying the optional arguments
    Parameters:
    -----------
    signal:         numpy 1d-array
                    noisy signal 
    beta:           numpy nd-array
                    Savitzky-Golay filter (computed via compute_filter)
    derivative:     int/list of ints, default [0]
                    order(s) of derivative to compute, has to be at most beta.shape[0] 
                    Each entry in the list will be the order of the derivative
                    e.g. derivative=[2,0,1], returns the second, 0th and first
                    derivative of the signal in each column
    stepsize:       float, default None
                    data grid width parameter (interval width), defaults to 1
                    if no derivative is requested
    Returns:
    --------
    smoothed_signal:    numpy nd-array
                        smoothed data, (default arguments)
                        If derivative is specified, shape is (len(signal, len(derivative) )
                        Each column refers to one of the specified derivatives
    """
    ## input check
    if isinstance( derivative, int):
        derivative = [derivative]
    if ( np.array( derivative) > 0).any():
        raise Exception( 'please specify stepsize to compute the derivatives' )
    else: #no derivative requested
        stepsize = 1

    ## preallocations
    window_length = beta.shape[1]
    pad_size = (window_length-1)//2
    #signal = pad. #TODO #pad the signal
    n       = len( signal)
    n_derivative = len( derivative)
    filtered = np.zeros( ( n, n_derivative  ) )

    ## Template using matrix multiplication with sparse matrices
    #for i in range( n_derivative ):
    #    smoothing = beta[#TODO]  #get the savitzky golay filter of the right differential order
    #    smoothing_array = sparse.diags( smoothing, offsets=range( -pad_size,pad_size+1) , shape=(n, n) ) #computes the convolution matrix as a sparse array
    #    filtered[:, i] = smoothing_array @ #TODO *factorial(derivative[#TODO])/... #TODO 


    #filtered = #TODO #unpad the signal
    return filtered

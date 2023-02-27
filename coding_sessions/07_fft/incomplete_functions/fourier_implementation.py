import numpy as np
from numpy import pi, exp


def DFT_brute_force( signal):
    """
    Computation of the DFT "by hand"
    Parameters:
    -----------
    signal:     1d-numpy array
                (preferably periodic) signal
    Returns:
    --------
    X:          1d-numpy array
                Fourier Transform of the 'signal'
    """
    N = signal.shape[0]
    X = np.zeros( signal.shape, dtype=complex)
    #for k in range( N ):
    #    for n in range( N ):
    #        X[k] += #TODO
    return X


def DFT( signal):
    """
    Computation of the DFT as a slightly more efficient implementation
    performs a matrix multiplication instead of double loops
    Parameters:
    -----------
    signal:     1d-numpy array
                (preferably periodic) signal
    Returns:
    --------
    X:          1d-numpy array
                Fourier Transform of the 'signal'
    """
    N = signal.shape[0]
    n_array = np.arange(N)
    ## preallocate the twiddle factors as matrix 
    ## and conduct the DFT with matrix multiplication
    #twiddle_factors = #TODO 
    #X = #TODO
    return X 


def FFT_naked( signal):#TODO possible other parameters
    """
    Your documentation goes here
    """
    # the FFT is a recursive divide and conquer algorithm
    # -> you will need to call the function here from within it
    # make sure that the recursion is resolved at some point with "if...return"
    # note that the DFT has to be computed on the odd and even split
    # since the function calls itself, no loops are required
    #TODO optional: write the FFT without any template (you can also use the result check)
    return











def FFT( signal, dft_size=32):
    """
    Computes the FFT with a recursive implementation of the Cooley-Tukey algorithm
    Parameters:
    -----------
    signal:     1d-numpy array
                (preferably periodic) signal
                signal length has to be a power of 2
    dft_size:   int, default 32 (2**5)
                on what recursive level the DFT should be computed
                has to be a power of 2
    Returns:
    --------
    X:          1d-numpy array
                Fourier Transform of the 'signal'
    """
    N = signal.shape[0] 
    if N % 2 > 0: # naively checks if our signals length is of the power of 2
        raise ValueError("size of the signal must be a power of 2")

    # when the signal is short enough, compute the DFT 
    # this is where the recursion is resolved (on the 'lowest level' of len(fft_split==dft_size) )
    if N == dft_size: 
        return #TODO #return the computed DFT

    else:
        ## splitting of the signal and recursive call on the even AND odd part
        #X_even = FFT( signal[#TODO],  dft_size)  
        #X_odd  = #TODO 

        ## weighting and assembly of both parts
        #omega = #TODO
        #left_part  = #TODO 
        #right_part = #TODO 
        #merged_signal = np.hstack( #TODO) #"stack/reassemble" the signal
        return #merged_signal


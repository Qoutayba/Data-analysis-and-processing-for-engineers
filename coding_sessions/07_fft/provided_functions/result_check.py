import numpy as np
from numpy import pi
from general_functions import tic, toc


def FT_implementation( FT_function, signal=None):
    """
    check if the fourier transform implementation is correct
    A signal can be specified to perform the FT on can be specified
    """
    if signal is None:
        signal = np.random.rand( 128)
    implementation = FT_function.__name__
    solution = np.fft.fftn( signal)
    tic( '{} implementation'.format( implementation), silent=True)
    if implementation == 'FFT_debug':
        entry = FT_function( signal, silent=True)
    else:
        entry = FT_function( signal)
    toc( '{} implementation'.format( implementation), precision=6 )
    if entry is None:
        raise Exception( 'Please implement the {} function'.format( implementation) )
    try:
        if np.allclose( solution, entry):
            print( "'{}' correctly implemented".format( implementation) )
            return
    except:
        raise Exception( 'wrong shape of signal returned' )

    ratios = np.array( [ solution[0]/entry[0], solution[-2]/entry[-2] ] )
    normalization = np.array( [1/np.prod(signal.shape),  (1/len(signal))**0.5 ])
    if np.allclose( ratios[0], normalization[0]) or np.allclose( ratios[0], normalization[1]):
        raise Exception( 'normalization possibly forgotten for the {}'.format( implementation) )
    if np.allclose( ratios[1], normalization[0]) or np.allclose( ratios[1], normalization[1]):
        raise Exception( 'normalization possibly forgotten for the {}'.format( implementation) )
    raise Exception( '{} wrongly implemented, got wrong values'.format( implementation) ) 



import numpy as np
import result_check as check


def zeroes( signal, pad_size):
    """
    Pad the given signal with zeros
    Parameters:
    -----------
    signal:         1d numpy-array
                    signal which should be padded
    pad_size:       int
                    size of the pad left AND right of the signal
    Returns:
    --------
    padded_signal:  numpy 1d-array
                    padded signal of size 2*pad_size + len(signal)
    """
    pad = np.zeros( pad_size)
    return np.hstack(( pad, signal, pad))



def periodic( signal, pad_size):
    """
    Pad the given signal by periodic continuation
    Parameters:
    -----------
    signal:         1d numpy-array
                    signal which should be padded
    pad_size:       int
                    size of the pad left AND right of the signal
    Returns:
    --------
    padded_signal:  numpy 1d-array
                    padded signal of size 2*pad_size + len(signal)
    Example: 
    -------- 
        padding.periodic( [1,2,3,4,5], pad_size=2)
        returns: [4,5, 1,2,3,4,5, 1,2]
    """
    #pad_left = #TODO
    #pad_right = #TODO
    return #TODO


def constant( signal, pad_size):
    """
    Pad the given signal by constnant continuation
    Parameters:
    -----------
    signal:         1d numpy-array
                    signal which should be padded
    pad_size:       int
                    size of the pad left AND right of the signal
    Returns:
    --------
    padded_signal:  numpy 1d-array
                    padded signal of size 2*pad_size + len(signal)
    """
    pad_left = np.ones( pad_size) *signal[0]
    pad_right = np.ones( pad_size) *signal[-1]
    return np.hstack(( pad_left, signal, pad_right))


def unpad( padded_signal, pad_size):
    """
    Unpad the given padded signal 
    Parameters:
    -----------
    padded_signal:  1d numpy-array
                    signal to un-pad
    pad_size:       int
                    size of the pad left AND right of the signal
    Returns:
    --------
    signal:         numpy 1d-array
                    signal of size len(padded_signal) - 2*pad_size 
    """
    # remove the previously appended "pads" with the "pad_size" variable
    # one liner with "slicing" possible
    #signal = padded_signal[ #TODO]
    return signal.squeeze() 


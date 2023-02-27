import numpy as np
import sys
import os
import time

initialized_times = dict()

def tic( tag='', silent=False):
    """
    initializes the tic timer
    different tags allow for tracking of different computations
    Parameters:
    -----------
    tag:        string, default ''
                name of tag to initialize the timer
    silent:     bool, default False
                Whether or not initialization should be printed
    """
    initialized_times[tag] = time.time()
    if not silent:
        print( 'Initializing timer for this tag:', tag)

def toc( tag='', precision=4 ):
    """
    prints the time passed since the invocation of the tic tag 
    does not remove the tag on call, can be timed multiple times 
    since start
    Parameters:
    -----------
    tag:        string, default ''
                name of tag to initialize the timer
    precision:  int, default 4
                How many digits after ',' are printed
    """
    time_passed = time.time() - initialized_times[tag]
    try:
        print( '{1} -> elapsed time:{2: 0.{0}f}'.format( precision, tag, time_passed) )
    except:
        print( 'tic( tag) not specified, command will be ignored!')




def file_size( filename):
    """
    Return the size of the given file in *iB format,
    Requires the convertBytes function
    Parameters:
    -----------
    filename:   string
                full path to file 
    Returns:
    --------
    byte_size:  float
                disc storage of the inspected file
    """
    if os.path.isfile(filename):
        file_info = os.stat(filename)
        return convert_bytes(file_info.st_size)


def convert_bytes( num):
    """
    Convert bytes into the largest possible format with num > 1
    The *iB refers to the binary file size format
    Parameters:
    -----------
    num:    float
            size of file given in machine format
            (e.g. 1040101049204 [bytes]
    Returns:
    --------
    num:    float
            converted size to human readable format
            (e.g. 1.04 TiB)
    
    """
    for x in ['bytes', 'KiB', 'MiB', 'GiB', 'TiB']:
        if num < 1024.0:
            return  "%3.2f %s" % (num, x)
        num /= 1024.0



def band_diagonal( n, width, value=1):
    """
    Generate a quadratic diagonal matrix with band structure, such that 
    the diagonal is of width 'width' e.g. [1,1,1, 0,0,0,0] in the 
    first row for width=5
    Parameters:
    -----------
    n:          int
                size of matrix
    widht:      int
                size of the diagonal, should be even
    value:      float, default 1
                value of the diagonal entries 
    """
    x = np.arange( n**2).reshape( n, n)
    counter = np.arange( n)
    diagonal_indices = np.abs( np.add.outer( counter, -counter) ) < width+1
    diagonal_matrix = np.zeros( ( n, n) )
    diagonal_matrix[ diagonal_indices] = value
    return diagonal_matrix


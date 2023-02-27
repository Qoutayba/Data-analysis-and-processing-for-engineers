import numpy as np
import time
import os

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


def file_size(fname):
    """
    return the size of the file "fname" in the ?iB format as string
    Parameters:
    -----------
    fname:      string
                absolute or relative location of the file to inspect
    Returns:
    --------
    size:       string
                size of the file with the corresponding unit
    """
    if os.path.isfile(fname):
        file_info = os.stat(fname)
        return convert_bytes(file_info.st_size)

def convert_bytes(num):
    """
    Convert bytes into the largest measure (i.e. kilo (k) mega(M), etc.)
    Gives it in *iB format, refering to binary file size format
    Parameters:
    -----------
    num:        int
                size in bytes
    Returns:
    --------
    size:       string
                converted size in ?iB 
    """
    for x in ['bytes', 'KiB', 'MiB', 'GiB', 'TiB']:
        if num < 1024.0:
            return  "%3.2f %s" % (num, x) 
        num /= 1024.0


def glorot_initializer( n_x, n_y=None):
    """
    Randomly initiazlize an array by the 'glorot_initializer'
    Parameters:
    -----------
    n_x:    int
            number of rows of the initialized array
    n_y:    int, default None, 
            number of colums of the initialized array
    Returns:
    --------
    random_array:       numpy nd-array
                        array filled with random numbers
                        1d if n_y is not specified
    """
    if n_y is None:
        limit = np.sqrt( 6) / n_x
        glorot = np.random.uniform( -limit, limit, size=(n_x) )
        return glorot
    else: 
        limit = np.sqrt( 6) / (n_x +n_y)
        glorot = np.random.uniform( -limit, limit, size=(n_x,n_y) )
    return glorot
 

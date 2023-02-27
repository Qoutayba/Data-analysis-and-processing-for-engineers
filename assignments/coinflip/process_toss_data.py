import h5py
import numpy as np

## This script serves as a reference for reading the submitted data

def int_to_str( tosses):
    """
    Format the integers of the tosses to human friendly string format.
    Changes every 0 to heads, and every 1 to tails.
    Parameters:
    -----------
    tosses:         list of ints
                    toss data made up of 0s and 1s
    Returns:
    --------
    string_tosses:  list of strings 
                    toss data given as 'heads' and 'tails' 
    """
    translation = ['heads', 'tails' ]
    translated_tosses = []
    for toss in tosses:
        translated_tosses.append( translation[toss] )
    return translated_tosses

## file loading and acces allocation
data = h5py.File( 'toss_data.h5', 'r' ) 
dataset = 'submission_{}'
n_submissions = len( data.keys() )

## print values of interest to console
inspected_submissions = np.random.randint( low=0, high=n_submissions, size=2)
for i in inspected_submissions:
    tosses = data[ dataset.format( i) ][:]
    print( '################## next submission ##################' )
    print( 'tosses for student nr {}:\n {}'.format( i+1, int_to_str( tosses) ) )
    print( 'Theta value of binomial distribution' )
    print( 'estimated:     {}'.format( np.mean( tosses) ) )
    print( 'vs analytical: {}'.format( 0.5 ) ) 
data.close()

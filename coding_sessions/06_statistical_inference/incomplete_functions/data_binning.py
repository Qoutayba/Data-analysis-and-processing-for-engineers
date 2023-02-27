import numpy as np

from numpy import pi, exp, sqrt
from math import ceil, floor

def bin_data( data, n_bins=None ):
    """
    Count the data and return the number of samples per bin
    Uni-length bins are set for simplicity
    If the number if n_bins is not specified it will automatically be computed
    Parameters:
    -----------
    data:       numpy 1d-array
                given data
    n_bins:     int, default None
                How many bins the data should be put into
                Computes a sensitive default value if not specified

    Returns:
    --------
    count:      numpy nd-array
                number of samples in the given bin
    center:     numpy nd-array
                center value of the given bin
    width:      float
                width of the bin
                bin start/end is center -/+ width/2
    """
    n_samples = data.shape[0] 
    # automatic choice for the number of bins
    if n_bins is None:
        if n_samples < 100:
            print( 'too few data samples present, returning un-binned data' )
            return [data.copy()]
        else:
            # Square-root choice
            n_bins = int(np.ceil(np.sqrt(n_samples)))
            
    ## Sort the data for easier handling
    data    = data[:].flatten() #create a copy as 1d-array
    sorting = np.argsort( data)
    data    = data[ sorting ] 

    ## Compute the bin centers and width
    lower_bound = data.min()
    #upper_bound = #TODO
    #stepsize    = #TODO 
    #center      = (0.5 + np.arange(n_bins)) * #TODO
    width       = stepsize

    ## Compute the number of samples in each bin
    count           = np.zeros(n_bins)
    previous_sample = 0 #which sample number the 'j' loop left off
    for i_bin in range( 0,n_bins-1):
        for j in range( previous_sample, n_samples):
            pass
            #if data[ j ] > #TODO
                #count[ i_bin] = #TODO
                #TODO... mark the j you left off and go to the next bin
    count[-1] = n_samples-count[0:-1].sum() #fill the last bin with the remaining samples
    return count, center, width

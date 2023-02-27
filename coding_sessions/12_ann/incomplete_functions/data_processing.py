import numpy as np
import h5py
from numpy.fft import fft2, ifft2
from math import ceil, floor


def split_data( inputs, outputs, split=0.3):  
    """ 
    Randomly shuffle the given data and split it into two sets (training and validation)
    Data has to be arranged column wise ( each column one sample)
    Parameters:
    -----------
    inputs:     nd numpy array
                array containing the input data
    outputs:    nd numpy array
                array containing the corresponding output data
    split:      float, default 0.3
                partition of the validation data 
    Returns:
    --------
    x_train, y_train, x_valid, y_valid:     nd-numpy arrays
                Input (x) and output (y) values in two sets
    """
    n_data  = inputs.shape[-1]
    #n_train = ceil(#TODO) 
    shuffle = np.random.permutation( n_data) #shuffle inputs and outputs accordingly
    #x_shuffled = #TODO 
    #x_train = #TODO #[..., slice] slices in the last axis
    #x_valid = #TODO
    y_shuffled = outputs[..., shuffle] #shuffle the samples (like a card deck)
    #y_train = #TODO  
    #y_valid = #TODO
    return x_train, y_train, x_valid, y_valid 


def scale_data( data, slave_data=None, scaletype='single_std1'):
    """
    Compute a shift based on data and apply it to slave_data
    Data has to be arranged column wise (each column one sample)
    Choose between the following scale methods:
    'single_std1': scale each component over all samples to have 0 mean and standard deviation 1
    'combined_std1': scale all component over all samples to have 0 mean and standard deviation 1 
    '-1-1': scale each component to lie on the interval [-1,1]
    '0,1': to be implemented, optional TODO
    Parameters:
    -----------
    data:           numpy nd-array
                    data to compute the shift/scaling
    slave_data:     numpy nd-array, default None
                    data to apply the shift to
    scaletype:      string, default 'single_std1'
                    specified scaletype to compute/apply
    Returns:
    --------
    data:       numpy nd-array
                shifted and scaled data
    slave_data: numpy nd-array, or None
                shifted and scaled slave data or None if no slave data was given
    scaling:    list of three
                Parameters and type of the scaling ( can be used in other functions)
    """ 
    n  = data.shape[-1]
    shift = [None,None, scaletype]
    
    if scaletype == 'single_std1':
        # shift the data to have 0 mean and a standard deviation of 1
        shift[0]   = np.mean( data, 1)[:,None]
        data       = data - shift[0]
        shift[1]   = np.sqrt( n-1) / np.sqrt( np.sum( data**2, 1))[:,None]
        data       = shift[1] * data 
    elif shift[2] == 'combined_std1':
        shift[0] = np.mean( data,1)[:,None]
        data     = data - shift[0]
        shift[1] = np.sqrt( data.size-1) / np.linalg.norm( data,'fro')
        data     = shift[1] * data 
    elif scaletype == '-1,1':  
        shift[0]   = np.min( data, 1)[:,None]
        #data       = #TODO 
        #shift[1]   = #TODO
        #data       = #TODO
    elif scaletype == '0,1':
        pass #TODO optional

    else:  #automatically returns unshifted data
        print( '########## Error Message ##########\nno valid shift specified, returning unshifted data and no scaling')
        print( "valid options are: 'single_std1', 'combined_std1', '-1,1', try help(scale_data)\n###################################" )
        shift[2] = None

    if slave_data is None:
        return data, shift 
    else:
        slave_data = scale_with_shifts( slave_data, shift )  
        return data, slave_data, shift 



def scale_with_shifts( data, shift ):
    """
    Apply the known shift computed with <scale_data()> to some data
    Parameters:
    -----------
    data:       numpy nd-array
                data arranged column wise 
    scaling:    list containing shift information
                previously computed scaling from <scale_data()>
    Returns:
    --------
    data:       numpy nd-array
                scaled data using 'scaling'
    """ 
    if shift[2] in ['single_std1', 'combined_std1'] :
        data = shift[1] * ( data - shift[0] ) 

    #elif shift[2] == '-1,1':
    #    data = #TODO #apply the shift stored in the list
    else: 
        print( 'scaletype not defined, returning raw data')
    return data


def unscale_data( data, shift ):
    """
    Takes previously shifted data and inverts the shift
    Shift must have been precomputed with <scale_data> and stored in a list 
    Parameters:
    -----------
    data    nd-numpy array
            shifted data aranged row-wise
    shift   list 
            contains the relevant data for the shift
    Returns:
    --------
    data    nd-numpy array
            original unscaled/unshifted data 
    """
    if shift[2] in ['single_std1', 'combined_std1']:
        return data / shift[1] + shift[0]
    elif shift[2] == '-1,1': 
        return #TODO #return the unscaled data
    else:
        print('scaletype not defined, returning raw data' )
        return data


def batch_data( x, y, n_batches, shuffle=True, stochastic=0.5):
    """
    Generator/Factory function, yields 'n_batches' batches when called as 
    a 'for loop' argument.  The last batch is the largest if the number of 
    samples is not integer divisible by 'n_batches' (the last batch is at 
    most 'n_batches-1' larger than the other batches)
    Also enables a stochastic chosing of the training samples by ommiting
    different random samples each epoch
    Parameters:
    -----------
    x:              numpy array
                    input data aranged column wise 
    y:              numpy array
                    output data/target values aranged column wise
    n_batches:      int
                    number of batches to return
    shuffle:        bool, default True
                    If the data should be shuffled before batching
    stochastic:     float, default 0.5
                    if the data should be stochastically picked, has to be <=1
                    only available if <shuffle> is True
    Yields:
    -------
    x_batch         numpy array
                    batched input data
    y_batch         numpy array 
                    batched output data
    """
    n_samples = y.shape[-1]
    if shuffle:
        permutation = np.random.permutation( n_samples )
        x = x[ ..., permutation] 
        y = y[ ..., permutation]
    else:
        stochastic = 0
    batchsize = int( n_samples // n_batches * (1-stochastic) )
    max_sample = int( n_samples* (1-stochastic) ) 
    i = -1 #to catch errors for n_batches == 1
    for i in range( n_batches-1):
        yield x[..., i*batchsize:(i+1)*batchsize], y[..., i*batchsize:(i+1)*batchsize]
    yield x[..., (i+1)*batchsize:max_sample ], y[..., (i+1)*batchsize:max_sample ] #last remaning samples


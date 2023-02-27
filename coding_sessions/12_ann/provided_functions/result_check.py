###################################################
# Data Processing for Engineers and Scientists
# Lectures by Prof. Felix Fritzen
# Computer labs by Julian Lissner
# 30.01.2020 - Lab 12 Artificial Neural Networks
###################################################
import numpy as np
import data_processing as process

def split_data():
    x = np.vstack( 2*[np.arange(15)] )
    y = x.copy()
    split = 0.333
    x_train, y_train, x_valid, y_valid = process.split_data( x, y, split)
    n_samples = x.shape[1]
    n_train = x_train.shape[1]
    n_valid = x_valid.shape[1] 
    returned_samples = n_train + x_valid.shape[1] 

    if  returned_samples != n_samples:
        raise Exception( 'Error in split_data():\n    data wrongly split, wrong total number of samples returned (expected {}, got {})'.format( n_samples, returned_samples) )
    if n_train <= n_valid:
        raise Exception( 'Error in split_data():\n    data wrongly split, expected approximately double training samples ({}) than validation samples ({})'.format( n_train, n_valid) )

    
    for i in range( n_train):
        if x_train[0,i] != x_train[0,i]:
            raise Exception( 'Error in split_data():\n    data has been wrongly shuffled, columns were unexpectedly shuffled' )
        if x_train[0,i] != y_train[0,i]:
            raise Exception( 'Error in split_data():\n    Outputs have been wrongly shuffled')
    print( 'Nice, "split_data" implemented correctly' )


def scalings( scaletype, transformation='full'):
    x = 9*np.random.rand( 20, 4).T
    test = 9*np.random.rand( 8, 4).T
    x_shifted, test_shifted, shift = process.scale_data( x, test, scaletype)

    min_val = np.round( np.min( x_shifted,1), 3)
    max_val = np.round( np.max( x_shifted,1), 3)

    if scaletype == '-1,1':
        if (min_val != -1 ).any(): 
            raise Exception( 'Error in scale_data():\n    expected a "-1" as the minimum in each feature, got\n     {} \n    instead'.format( min_val) )
        if (max_val != 1 ).any(): 
            raise Exception( 'Error in scale_data():\n    expected a "1" as the maximum in each feature, got\n     {} \n    instead'.format( max_val) )
        if transformation != 'full':
            print( 'Nice, forward transform correctly implemented' )


    if transformation == 'full':
        ## first check if scale with shift is correctly implemented
        scale_test = process.scale_with_shifts( x, shift)
        if (not np.allclose( scale_test.shape, x_shifted.shape)) or (not np.allclose( scale_test, x_shifted)):
            raise Exception( 'Error in scale_with_shifts():\n   slave data was wrongly shifted')
        ## then check if the unscaling is correctly implemented
        x_orig = process.unscale_data( x_shifted, shift) 
        test_orig = process.unscale_data( test_shifted, shift)
        if x_orig is None and test_orig is None:
            raise Exception( 'please implement the "data_processing.unscale_data" function for scaletype: {}'.format( scaletype) )

        if not np.allclose( x, x_orig):
            raise Exception( 'Error in unscale_data():\n    back transformation did not yield the original data (for the data the shifts were computed with)')
        if not np.allclose( test, test_orig):
            raise Exception( 'Error in unscale_data():\n    back transformation did not yield the original data (for the slave data)')
        if scaletype != '-1,1':
            print( 'only forward-backwardtransform has checked and is correct, scaled data not investigated, no guarantee that "{}" is implemented correctly!'.format( scaletype) )



    

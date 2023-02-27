import numpy as np
import pickle
import padding as pad
import filters 

signal   = np.array( [1,2,3,4,5] )
pad_size = 2


def periodic_padding():
    """
    check if periodic padding is correctly implemented
    by directly calling the function
    """
    result          = np.array([4,5, 1,2,3,4,5, 1,2])
    entry           = pad.periodic_padding( signal, pad_size)
    try:
        match = np.allclose( result.shape, entry.shape)
        if not match:
            padding_error( 'periodic padding', result, entry )
    except:
        padding_error( 'periodic padding', result, entry )
    if not np.allclose( result, entry):
        padding_error( 'periodic padding', result, entry )
    print( 'periodic padding correctly implemented' )



def un_padding():
    """
    check if un-padding is correctly implemented
    by directly calling the function
    """
    result          = np.array([1,2,3,4,5])
    padded_signal   = np.array([4,5, 1,2,3,4,5, 1,2])
    entry           = pad.unpad_signal( padded_signal, pad_size=2)
    try:
        match = np.allclose( entry.shape, result.shape)
        if not match:
            padding_error( 'unpad signal', entry, result )
    except:
        padding_error( 'unpad signal', entry, result )
    if not np.allclose( entry, result ):
        padding_error( 'unpad signal', entry, result )
    try:
        if not np.allclose( pad.unpad_signal( result, pad_size=0 ), result):
            raise Exception( 'wrongly unpadded for pad_size=0' )
    except:
            raise Exception( 'wrongly unpadded for pad_size=0' )
    print( 'unpad signal correctly implemented' )



def padding_error( function_name, result, entry):
    """ an error template when a padding error occured"""
    error_message = """'{}' wrongly implemented\n
    desired signal: \t{}
    obtained signal:\t{}""".format( function_name, result, entry) 
    raise Exception( error_message) 


########## Filter functions
def moving_average( filter_function):
    """check the implementation of any moving average function """
    np.random.seed( 69)
    signal = np.random.randint( 0, 7, size=(7) )
    entry   = filter_function( signal, 3)

    try:
        results = np.load( 'data/mean_filter_results.npz')
    except:
        results = np.load( '../data/mean_filter_results.npz')
    result_periodic  = results ['periodic']
    result_const  = results ['const']
    result_zero  = results ['zero' ]
    result = list( result_zero)
    result[0] = '*'
    result[-1] = '*'

    dimension_mismatch = ('moving average wrongly implemented, expected signal of shape'
                        + ' {}, got {} instead').format( result_const.shape, entry.shape)
    try:
        match = np.allclose( result_zero.shape, entry.shape )
        if not match:
            raise Exception( dimension_mismatch)
    except:
        raise Exception( dimension_mismatch)
    error_message = ("'moving average' wrongly implemented test with window length 3\n"
     + '(edges omitted because of user-defined padding)'
     + '\tcheck signal:  \t{}\n'
     + '\tdesired result:\t{}\n'
     + '\tcomputed result:\t{}').format( signal, result, entry)
    if not ( np.allclose( entry, result_const) or np.allclose( entry, result_zero)
        or np.allclose( entry, result_periodic) ):
        raise Exception( error_message)
    original = filter_function( entry, window_length=1)
    if not np.allclose( original, entry):
        raise Exception( 'Error for window length 1, make sure the neighbourhood is considered correctly')
    print( '{} correctly implemented'.format( filter_function.__name__ ) )



def precompute_savgol( poly_order=2, window_length=None ):
    _ = filters.compute_filter( poly_order, window_length, check_flag=True)
    

def savgol_filter( J, sav_gol, poly_order, window_length, check_flag ):
    if check_flag is False:
        return
    try:
        f = open( 'data/sav_gol.pkl', 'rb')
    except:
        f = open( '../data/sav_gol.pkl', 'rb')
    results = pickle.load( f)
    f.close()
    #contains a list of nested lists of lists containing nd-arrays
    try:
        J_result       = results[0][poly_order][window_length].round(9)
        sav_gol_result = results[1][poly_order][window_length].round(9)
    except: 
        print( 'defined savitky golay filter of unprepared input parameters.\n No check conducted\t ...continuing')
        return
    errors = []
    if not isinstance( J_result, np.ndarray):
        raise Exception( "Corrupt inputs for 'poly_order' or 'window_length', window_length needs to be larger than poly_order")
    slice_J = tuple( [ slice(0, J.shape[i]) for i in range(J.ndim) ] ) 
    if not (J_result[slice_J]  == J.round(9)).all():
        errors.append( 'J')
    slice_savgol = tuple( [ slice(0, sav_gol.shape[i]) for i in range(sav_gol.ndim) ] )
    if not (sav_gol_result[slice_savgol] == sav_gol.round(9)).all():
        errors.append( 'sav_gol/convolution coefficients')
    if errors:
        raise Exception( "Some arrays contain errors, please fix:\t {}".format( errors) )
    return #everything correct



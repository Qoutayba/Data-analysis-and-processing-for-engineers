import sys
import numpy as np

def compare_arrays( **arrays):
    """
    Compare the given arrays as kwargs with pre-stored arrays 
    stored in the 'data/arrays.npz' file 
    If an unknown array is given as input argument, 
    it will also be displayed in the error message.
    Catching an error in any of these arrays will raise an exception
    """
    results = np.load( 'data/results.npz' )
    available_results = list( results.keys() )
    wrong_values = []
    wrong_shapes = []
    wrong_argument = []
    ## loop over every given named array and check if it matches the solution
    for key in arrays:
        if key not in available_results:
            wrong_argument.append( key)
            continue
        try: 
            if not np.allclose( results[key], arrays[key] ):
                wrong_values.append( key)
        except:
            wrong_shapes.append( key)
            continue
        if not np.allclose( results[key].shape, arrays[key].shape ):
            wrong_shapes.append( key)
    error_message = 'Error in the current task:\n'
    something = True
    wrong = False
    ## If there is any error, give specific console output based off the error
    if wrong_values: 
        wrong = True
        error_message = error_message + 'Values of these matrices do not match:\n '
        for error in wrong_values:
            error_message = error_message + error + ',   '
        error_message = error_message[:-4]

    if wrong_shapes: 
        wrong = True
        error_message = error_message + '\nshapes of these matrices do not match:\n'
        for error in wrong_shapes:
            if error in 'umhds': #scalar value expected
                error_message = error_message + ' {}: got {}, expected a scalar\n'.format( error, arrays[error].shape )
            else:
                error_message = error_message + ' {}: got {}, expected {}\n'.format( error, arrays[error].shape, results[error].shape )

    if wrong_argument:
        wrong = True
        error_message = error_message + '\nCould not compare the following arrays (not known in solution):\n '
        for error in wrong_argument:
            error_message = error_message + error + ',   '
        error_message = error_message[:-4]

    if something is wrong:
        raise Exception( error_message )
    print( 'Part correctly solved.')

    




import pickle
import numpy as np

def data_binning( bin_occurences, bin_centers, bin_width, n_bins):
    """ check if the data has been correctly binned) """
    if n_bins >= 50:
        print( 'Too many bins specified for the data, result check can not be conducted' )
        return
    if len( bin_occurences) != n_bins or len( bin_centers ) != n_bins:
        raise Exception( 'data has been wrongly binned, length of "bin_occurences" and "bin_centers" has to match n_bins' )
    results = pickle.load( open( 'data/bin_results.pkl', 'rb' ))
    errors = []
    if not np.allclose( results['occurences'][n_bins], bin_occurences ):
        errors.append( 'bin_occurences wrongly computed. Make sure bin_occurences.sum() == n_samples\n' )
    if not np.allclose( results['centers'][n_bins], bin_centers ):
        errors.append( 'bin_centers wrongly computed. Make sure that the bin centers do not start at data[0]\n' )
    if not isinstance( bin_width, float):
        print( 'Warning, bin_width is a single float in the result check, conducting simple check')
        if not np.allclose( results['widths'][n_bins], np.mean( bin_width ) ):
            print( 'possible error in bin_width' )
    else:
        if not np.allclose( bin_width, results['widths'][n_bins] ):
            errors.append( 'bin_width wrongly specified' )

    if errors:
        raise Exception( 'Errors found in "bin_data":\n' +(len(errors)*'{}').format( *errors ) )
    print( 'data correctly binned' ) 

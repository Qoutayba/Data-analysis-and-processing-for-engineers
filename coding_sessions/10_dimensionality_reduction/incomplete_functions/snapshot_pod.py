import numpy as np
import result_check as check

def truncation( sigma, truncation_threshold):
    """
    Compute the truncation limit given the singular values/ root of eigenvalues
    Parameters:
    -----------
    sigma:                  1d-numpy array
                            array of singular values/root of eigenvalues 
                            (sorted in descending order)
    truncation_threshold:   float
                            truncation threshold (governs mean projection relative error)
    Returns:
    --------
    N:          int
                smallest number of eigenmodes to achieve the specified error
    """
    info = np.sum( sigma**2 ) #efficiency allocation
    for N in range( len(sigma) ):
        #error = np.sqrt( np.max( (0,#TODO ) ) ) #TODO
        if error <= truncation_threshold:
            break
    return N

def correlation_matrix( snapshots, truncation_threshold, metric=None):
    """
    Compute the reduced basis via the 'snapshot correlation matrix'
    Parameters:
    -----------
    snapshots:              numpy 2d-array
                            column wise aranged snapshot data (each column one sample) 
    truncation_threshold:   float
                            truncation threshold for basis computation
    metric:                 numpy 2d-array, default None
                            metric with which the basis is computed
                            defaults to no metric (identity matrix)
    Returns:
    --------
    reduced_basis:          numpy array
                            truncated reduced basis computed of <snapshots>
    """
    if metric is None:
        sigma, Q = np.linalg.eigh( snapshots.T @ snapshots)
    #else: #a metric is given
    #    sigma, Q = np.linalg.eigh( #TODO ) 
    #sigma = #TODO #note: eigh returns eigenvalues in ascending order
    #Q     = Q[#TODO  #sort Q correspondingly to sigma
    N     = truncation( sigma, truncation_threshold ) 
    sigma = sigma[:N]
    #TODO...
    return #TODO

def svd_rb( snapshots, truncation_threshold, metric=None):
    """
    Compute the reduced basis via the 'singular value decomposition'
    Parameters:
    -----------
    snapshots:              numpy 2d-array
                            column wise aranged snapshot data (each column one sample) 
    truncation_threshold:   float
                            truncation threshold for basis computation
    metric:                 numpy 2d-array, default None
                            metric with which the basis is computed
                            defaults to no metric (identity matrix)
    Returns:
    --------
    reduced_basis:          numpy array
                            truncated reduced basis computed of <snapshots>
    """
    if metric is None:
        B, sigma = np.linalg.svd( snapshots, full_matrices=False)[:-1]
        N = truncation( sigma, truncation_threshold)
        return B[:,:N]
    #else: 
        #L = TODO
        return #TODO



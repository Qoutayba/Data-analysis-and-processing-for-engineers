import numpy as np
import matplotlib.pyplot as plt
import snapshot_pod as pod
from numpy.linalg import norm


def image_truncation( N, V, sigma, WT, reconstructed_image, image):
    """ Check if the the truncation has been correctly conducted 
    and if the image has been correctly reassembled for svd image compression """
    shape = image.shape
    errors = ''
    if len( sigma) != N:
        errors = errors + '"sigma" wrongly truncated, expected {} singular values, got {}\n'.format( N, len(sigma) ) 
    if V.ndim == 1 or WT.ndim ==1:
        errors = errors + 'V or WT sliced to vector, expected a truncated matrix\n'
    if not np.allclose(V.shape, ( shape[0], N)):
        expected_shape = ( shape[0], N)
        errors = errors + '"V" wrongly truncated, expected the following shape {}, got {}\n'.format( expected_shape, V.shape ) 
    if not np.allclose( WT.shape, (N, shape[1] ) ): 
        expected_shape = ( N, shape[1] )
        errors = errors + '"WT" wrongly truncated, expected the following shape {}, got {}\n'.format( expected_shape, WT.shape ) 
    if not np.allclose( reconstructed_image.shape, shape):
        errors = errors + 'image wrongly reassembled, got shape: {}, expected: {}'.format( reconstructed_image.shape, shape )
    try:
        if not errors and not np.allclose( reconstructed_image, V*sigma@WT ):
            errors = errors + 'image wrongly reassembled, unexpected values found in image'
    except:
        if not errors and np.allclose( reconstructed_image, V@sigma@WT ):
            errors = errors + 'image wrongly reassembled, unexpected values found in image'
    if errors:
        raise Exception( errors)



def truncation_implementation():
    """ check the truncation implementation by looking 'left and right' of the computed threshold"""
    data = np.random.rand( 500, 5000)
    aaa = np.linalg.svd( data, full_matrices=False)
    truncation_threshold = 0.05
    N = pod.truncation( aaa[1], truncation_threshold )
    if N is None:
        raise Exception ('Please implement the truncation function, "None" was returned')
    data_norm = np.linalg.norm(data,'fro')
    xi = aaa[0].T @ data
    compute_e = lambda xi, N, data_norm:   np.sqrt( 1- (np.linalg.norm(xi[:N,:], 'fro')/data_norm)**2)#1- np.sqrt(1 - np.linalg.norm( xi[:,:N],'fro')**2/data_norm**2) 
    error =  compute_e( xi, N, data_norm )
    prev_error = compute_e( xi, N+2, data_norm)
    next_error = compute_e( xi, N-2, data_norm)
    if not (prev_error <= error <= truncation_threshold <= next_error):
        raise Exception( 'Error in "truncation", expected error (approximately): {:.5f}, instead got: {:.5f}'.format( truncation_threshold, error) )
    

def reduced_basis( compute_rb):
    """ Compare the reduced basis to a precomputed RB and analyze the errors"""
    basis = np.load( 'data/reduced_basis_check.npz')
    np.random.seed(69)
    data = np.random.rand( 64, 40)
    metric = np.random.rand( 64,64)
    metric = metric.T @ metric
    trunc = 0.01 
    error_msg = """ these variables might help debugging:\n
expected frobenius norm\t{}\n given frobenius norm \t{}\n\n
expected shape of the basis\t{}\n given shape of the basis\t{}\n\n
expected diagonal structure of B.T @ (M) @ B = I, given 'structure'\n{}"""

    #first check without the metric
    error_header = 'error in {} (without metric M),'.format( compute_rb.__name__) 
    B_true = basis['arr_0']
    B_norm = norm(B_true, 'fro')
    B = compute_rb( data, trunc )
    if B is None:
        raise Exception( error_header + 'returned None, please fix the return statement' )
    elif B.ndim ==1:
        raise Exception( error_header + 'the RB was returned as a vector, expected a matrix of dimension n_s x N ')
    for i in range( B.shape[1]):
        if i >= B_true.shape[1]:
            break
        if B[:,i].T @ B_true[:,i] < 0.8:
            B[:,i] = -B[:,i]
    if (not np.allclose( B.shape, B_true.shape)) or (not np.allclose( B, B_true)):
        raise Exception( error_header + error_msg.format( B_norm, norm( B,'fro'), B_true.shape, B.shape, (B.T@B)[:4,:4].round(6) ) )

    #check with the metric
    error_header = 'error in {} (using metric M),'.format( compute_rb.__name__)
    B_true = basis['arr_1']
    B_norm = norm(B_true, 'fro')
    B = compute_rb( data, trunc, metric )
    error_header = 'error in {},'.format( compute_rb.__name__)
    if B is None:
        raise Exception( error_header + 'returned None, please fix the return statement' )
    elif B.ndim ==1:
        raise Exception( error_header + 'the RB was returned as a vector, expected a matrix of dimension n_s x N ')
    for i in range( B.shape[1]):
        if B[:,i].T @ metric @ B_true[:,i] < 0.8:
            B[:,i] = -B[:,i]
    if (not np.allclose( B.shape, B_true.shape)) or (not np.allclose( B, B_true)):
        raise Exception( error_header + error_msg.format( B_norm, norm( B,'fro'), B_true.shape, B.shape, (B.T@ metric @B)[:4,:4].round(6)) )





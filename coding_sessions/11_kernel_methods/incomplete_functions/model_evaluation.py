import numpy as np
from result_check import gaussian_kernel

def classification( sample, x_negative, x_positive, weight_negative, weight_positive, bias, gamma, classify=True):
    """ 
    Evaluate the trained model for a single sample.
    Compute the evaluation y_hat for the sample and return the classification {-1, +1}
    Parameters:
    -----------
    sample:             numpy nd-array
                        sample to evaluate
    x_negative:         numpy nd-array
                        support points with negative output value
    x_positive:         numpy nd-array
                        support points with positive output value
    weight_negative:    numpy nd-array
                        weight computed for x_negative
    weight_positive:    float
                        weight computed for x_positive
    bias:               float
                        precomputed bias of the model
    gamma:              float
                        hyperparameter for the gaussian kernel
    classification:     bool, default True
                        if the evaluation should be classified
                        THIS IS REQUIRED FOR PLOTTING, DO NOTE REMOVE THIS INPUT
    Returns
    ------- 
    classification:     float
                        result for the sample, either +1 or -1
    """
    #value  = #TODO #model evaluation using the passed parameters
    #value += #TODO...
    if not classify: #required for the plotting function
        return value
    if value < 0:
        return #TODO #classification
    else:
        return #TODO 

def interpolation( x_train, x_prime, kernel_function, parameters, coefficients=None, y_train=None, nugget=1e-10):
    """ 
    Intervaluate a trained model for interpolation/regression. If y_train
    is given, a model for interpolation is computed.  If no <coefficients>
    are given, then they are calibrated with the given training data
    (<y_train> needs to be specified).
    To use this function for regression, the <coefficients> need to be passed.
    Parameters:
    -----------
    x_train:            numpy nd-array
                        support points of the model
    x_prime:            numpy nd-array
                        points where the model should be evaluated
    kernel_function:    function handle
                        function handle to the kernel function
    parameters:         list of floats
                        additional parameters taken by the <kernel_function>
    coefficients:       numpy nd-array, default None
                        coefficients required for interpolation multiplication
                        if they are not given, they will be computed
                        the computation requires specification of y_train
    y_train:            numpy nd-array, default None
                        ouput value of <x_train>
    nugget:             float, default 1e-10
                        regularization parameter to circumvent singularities 
                        only required if <coefficients> is None
    Returns:
    --------
    y_prime:            numpy nd-array
                        evaluation of the interpolation for <x_prime> using
                        the model derived from <x_train>
    """
    if coefficients is None: #if no w given, compute it
        if y_train is None:
            raise Exception( 'error in "model_evaluation.interpolation": y_train required if coefficients are computed')
        #kernel_matrix = #TODO
        #kernel_matrix -= nugget *  #TODO
        #K_inv =  #TODO
        #coefficients = #TODO

    kernel_response = kernel_function( x_train, x_prime, *parameters)
    #y_interpolated = #TODO
    return y_interpolated

import numpy as np

def checkerboard_reference( resolution=(100,100), n_x=4, n_y=4 ):
    """ 
    Get the reference solution of the checkerboard task
    Parameters:
    -----------
    resolution:     list like of two ints, default (100,100)
                    resolution of the resulting checkerboard
    n_x:            int, default 4
                    number of checkers in x direction (vertical)
    n_y:            int, default 4
                    number of checkers in y direction (horizontal) 
    Returns:
    --------
    checkerboard:   numpy 2d-array
                    array of the checkerboard best seen as an image
    """
    counter = np.outer( np.arange( resolution[0]), np.arange( resolution[1]) )  
    dx = int(resolution[0]/n_x )
    dy = int(resolution[1]/n_y )
    checkerboard = np.ones( resolution) 
    for i in range(n_x):
        for j in range(n_y):
            sign = (-1)**(i+j)
            if i==(n_x-1) and j != (n_y-1):
                checkerboard[ i*dx:, j*dy: (j+1)*dy-1] *=sign
            elif i !=(n_x-1) and j == (n_y-1):
                checkerboard[ i*dx: (i+1)*dx-1, j*dy: ] *= sign
            elif i == (n_x-1) and j == (n_y-1):
                checkerboard[ i*dx: , j*dy: ] *= sign
            else:
                checkerboard[ i*dx: (i+1)*dx-1, j*dy: (j+1)*dy-1] *= sign
    return checkerboard


def checkerboard( checkerboard, n_samples, n_noisy=None):
    """ use the checkerboard reference to sample n_samples of the checkerboard"""
    if n_noisy is None:
        n_noisy = int( 0.05*n_samples) # 5% noise
    y = np.zeros( n_samples)
    resolution = checkerboard.shape
    support_points = np.vstack( (np.random.randint( 0, resolution[0], size=(n_samples) ),
                                np.random.randint( 0, resolution[0], size=(n_samples) )) ).T
    for i in range( support_points.shape[0] ):
        y[i] = checkerboard[ tuple(support_points[i,:]) ] 
    noisy_samples = np.random.randint( n_samples, size=(n_noisy) )
    y[ noisy_samples] *= -1 #get some wrong values as <noise>
    # scale the support points (x) to float values
    support_points = support_points / np.array( resolution)
    return support_points, y



def function( function, interval, n_samples, noise=None):
    """
    Sample the given function randomly on the interval.
    some noise is added per default, can be turned off if noise=0
    Parameters:
    -----------
    function:   function handle
                function to sample from which takes 
                only 1 numpy 1d-array as required input
    interval:   list of 2 floats
                lower and upper limit of the defined interval
    n_samples:  int
                number of samples to return
    noise:      float, default None
                magnitude of the noise (normal distributed)
                if None is given, it defaults to '0.05*f(x).max()'
    Returns:
    --------
    x:          numpy 1d-array
                sampled points in the interval
    y:          numpy 1d-array
                evaluation of the function of samples x 
    """
    x = np.random.uniform( *interval, size=n_samples)
    y = function( x)
    if noise is None:
        noise = np.random.randn( *x.shape)
        noise = noise/noise.max() * y.max() *0.05
    else:
        noise = np.random.randn( *x.shape) * noise
    y += noise
    return x, y



def uniform_interval( lower_bound, upper_bound, n_steps=200 ):
    """
    Sample the interval defined by lower and upper bound in 
    equidistant steps
    Parameters:
    -----------
    lower_bound:    float
                    lower bound of the interval
    upper_bound:    float
                    upper bound of the interval
    n_steps:        int
                    number of samples to return
    Returns:
    --------
    x:          numpy 1d-array
                sampled points in the interval
    """
    x = np.arange( 0, 1, 1/n_steps)
    x = x*(upper_bound-lower_bound) + lower_bound
    return x

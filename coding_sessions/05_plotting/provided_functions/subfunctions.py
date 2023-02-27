import numpy as np
from math import ceil, floor
from sklearn import pipeline, preprocessing, linear_model


def regression( x, y, poly_order ): 
    """
    Fit a polynomial regression to the given data and return a sampled
    array which spans the whole interval of x and the respective values
    of y (regression) in the interval
    Paramters:
    ----------
    x:          numpy 1d-array
                input value the given samples
    y:          numpy 1d-array
                target value the given samples
    poly_order: int
                polynomial order of the regression
    Returns:
    --------
    x:          numpy 1d-array
                range sampled over the interval of x
    y:          numpy 1d-array
                regression fit over the whole interval x
    """
    regression = pipeline.make_pipeline( preprocessing.PolynomialFeatures( poly_order), linear_model.LinearRegression())
    regression.fit( x.reshape(-1,1), y.reshape(-1,1) )
    if x.min() < 0:
        xmin = 1.2* x.min()
    else:
        xmin = 0.8* x.min()
    x_range = np.arange( xmin, 1.2*x.max(), x.max()/250 )
    y_range = regression.predict( x_range.reshape(-1,1) ).flatten()
    return x_range, y_range


############## FUNCTIONS REQUIRED FOR THE EXAMPLE PLOTS ################
########################################################################
def get_min_vals( array):
    """
    Find the minimum location of x for minima in y
    Parameters:
    -----------
    array:      numpy nd-array
                column wise arranged data of y values
    Returns:
    --------
    xmin:       numpy 1d-array
                index position of the minimum
    ymin:       numpy 1d-array
                value of the minimum
    """
    ymin = np.min( array, axis=0 ) 
    x = np.arange(array.shape[0])
    xmin = []
    for minima in (array==ymin).T: # array needs to be transposed for the loop
        try:
            xmin.append( x[minima]  )
        except: #if there are multiple same minima
            xmin.extend( x[minima]  ) 
    return xmin, ymin


def compute_sample_bounds( x, y, stepsize=100 ):
    """
    given sampled data y(x), compute the bounds (mean, max and min values) of y(x)
    Gives three arrays of length stepsize which is e.g. min( y(x') ), with x' replicating the given x
    Parameters:
    -----------
    x:          numpy 1darray
                sample input parameters
    y:          numpy 1darray 
                sampled values y(x), length of y and x must match 
    stepsize:   int, default 100
                in how many increments the minimum and maximum values should be displayed
    Returns:
    --------
    x':             numpy 1darray
                    virtuell input parameter for the computed bounds
    min_bound:      numpy 1darray
                    minimum bounds of the sampled data
    mean_bound:     numpy 1darray
                    mean bounds of the sampled data
    max_bound:      numpy 1darray
                    maximum bounds of the sampled data
    #NOTE that i might be able to modifiy the function such that y is a nd-array
    """
    permutation = x.argsort()
    x = x[permutation]
    y = y[permutation]
    increment = (max(x)-min(x))/stepsize
    x_virt = np.arange( min( x), max(x)+increment, increment  )[:stepsize] #slice again that the array is never too big
    x_virt = x_virt 
    min_bound = np.zeros( stepsize)
    max_bound = np.zeros( stepsize)
    mean = np.zeros( stepsize)
    n = y.shape[-1]
    step = n/stepsize
    min_bound[0] = min( y[0 : ceil( step) ] )
    max_bound[0] = max( y[0 : ceil( step) ] )
    mean[0] = np.mean( y[0 : ceil( step) ] )
    for i in range( 1, stepsize):
        lower_slice = max( 0, floor( step*(i-1) ) )
        upper_slice = max( 1, ceil( step*(i)) )
        min_bound[i] = min( y[ lower_slice:upper_slice])
        max_bound[i] = max( y[ lower_slice:upper_slice])
        mean[i]      = np.mean( y[ lower_slice:upper_slice])
    return x_virt, min_bound, max_bound, mean



def layout( ax, xlabel=None, ylabel=None, title=None, grid=True, **legend_kwargs):
    """ 
    Add some style layout to the plot if specified
    Parameters:
    -----------
    ax:             matplotlib axes object
                    ax to put the style in
    xlabel:         string, default None
                    label of x-axis
    ylabel:         string, default None
                    label of y-ayis
    title:          string, default None
                    title of the plot
    grid:           bool, default True
                    add specified grid (dashed, lw=1.5)
    legend_kwargs:  dict
                    style of the added legend
    Returns:
    --------
    ax:             matplotlib axes object
                    axes object with added style,
                    does not have to be caught
    """
    #set the xlabel, ylabel and title using the input arguments
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title) 
    # add a grid to the plot and give it a default style (dashed lines, lw=1.5)
    if grid is True:
        ax.grid( ls='--', lw=1.5) 
    if legend_kwargs: #checks if arguments have been passed
        #use the 'unpacking operator' ** to give the "legend()" function its arguments
        key = ax.legend( **legend_kwargs )
        key.get_frame().set_linewidth( 2.5 ) #set a thicker linewidth 
    return ax 

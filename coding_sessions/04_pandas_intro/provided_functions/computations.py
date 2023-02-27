import numpy as np
from sklearn import pipeline, preprocessing, linear_model




def regression( x, y, poly_oder=2):
    """
    Compute the regression of defined order for the given data
    Parameters:
    -----------
    x:          numpy 1d-array
                x- component of the data
    y:          numpy 1d-array
                y- component of the data
    poly_order: int, default 2
                polynomial order of the regression
    Returns:
    --------
    x_sampled:  numpy 1d-array
                x-values for the line being slightly spanning a slightly
                larger interval than [x.min(), x.max() ]
    y_sampled:  numpy 1d-array
                y-values of the regression calculated by the given points
    """
    regression = pipeline.make_pipeline( preprocessing.PolynomialFeatures( degree_reg),
                                        linear_model.LinearRegression())
    regression.fit( x.reshape(-1,1), y.reshape(-1,1) )

    if x.min() < 0:
        x = np.arange( 1.2*x.min(), 1.2*x.max(), x.max()-x.min()/300 ).reshape( -1,1)
    else:
        x = np.arange( 0.8*x.min(), 1.2*x.max(), x.max()-x.min()/300 ).reshape( -1,1)
    return x, regression.predict( x).flatten()


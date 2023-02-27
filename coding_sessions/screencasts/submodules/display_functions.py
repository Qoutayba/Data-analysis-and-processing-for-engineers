import numpy as np
import matplotlib.pyplot as plt


def underlying_distribution(x, bins=10):
    """
    Returns values for a plot of the sampled distribution "x".
    This simulates the probability density of the given data 
    Parameters:
    -----------
    x:      numpy nd-array
            data given as sample values
    bins:   int, default 1--
            number of discretized bins to interpolate the plot with
    Returns:
    --------
    x:      numpy nd-array
            interval over which y is defined
    y:      numpy nd-array
            distribution function of x
    delta_x:float
            virtual bin width
    """
    x = np.squeeze(x) #x is required to be of shape (n,)
    x = np.sort(x)
    n = len(x)
    bins = min(n//3, bins) #ensures that there are not too rough jumps
    f = np.zeros( (bins) )
    delta_x = (x[-1]-(x[0]) )/bins
    scalefactor = ((x[-1]-x[0])/(bins)) *n
    for i in range(bins):
        x_bin = x[0]+i*delta_x
        f[i] = (np.searchsorted(x, max(x_bin+delta_x,x[0]) ) - np.searchsorted(x, min(x_bin,x[-1]) ) )/scalefactor

    x_dist = np.arange(x[0], x[-1], (x[-1]-x[0])/(bins))[:bins]
    x_dist = x_dist + (x_dist[1]- x_dist[0])/2 #moves each value in the center of the 'bin', not the beginning

    return x_dist, f, delta_x


def cumulative_distribution_function( x):
    """
    Return values for a plot of the cumulative distribution function
    Parameters:
    -----------
    x:      numpy nd-array
            data given as sample values
    Returns:
    --------
    x:      numpy nd-array
            interval over which y is defined
    y:      numpy nd-array
            distribution function of x
    """
    x = x.squeeze().copy() #make sure that the passed x is not altered
    x.sort()
    n = len(x)
    y = (np.arange( n)+1 )/n
    return x, y



def convolution_examples():
    """ plot the examples for the convolution as 'image' 'kernel' 'result' """
    images    = list( np.load( 'data/example_convolutions.npz' ).values() )
    fig, axes = plt.subplots( 3,3, figsize=(12,12))
    for ax in axes[:,0]:
        ax.imshow( images[2])

    axes[0,1].imshow( images[-6])
    axes[1,1].imshow( images[-4])
    axes[2,1].imshow( images[-2], cmap='gray')

    axes[0,2].imshow( images[-5].astype(float))
    axes[1,2].imshow( images[-3].astype(float))
    axes[2,2].imshow( images[-1].astype(float), cmap='gray')

    for ax in axes[:,[0,2]].flatten():
        ax.axis( 'off')
    axes[1,0].axis( 'on')
    axes[0,0].set_title( 'Image' )
    axes[0,1].set_title( 'Convolution Kernel' )
    axes[0,2].set_title( 'Image * Kernel' )
    plt.show()


def unpadded_convolution():
    images    = list( np.load( 'data/convolution_comparison.npz' ).values() )
    titles = [ 'Image', 'convolution' ] 
    fig, axes = plt.subplots( 1,2, figsize=(10,5))
    for i in range( 2 ):
        axes[i].imshow( images[i].astype(float))
        axes[i].set_title( titles[i] )
        axes[i].axis( 'off' )
    plt.show()


def padding_examples():
    """ plot the comparison of periodic and non periodic convolution as image - normal - periodic convoluition """
    images    = list( np.load( 'data/convolution_comparison.npz' ).values() )
    titles = [ 'image', 'no padding', 'zero padding', 'periodic padding' ]
    fig, axes = plt.subplots( 2,2, figsize=(12,10))
    axes = axes.flatten()
    for i in range(len( axes) ):
        axes[i].imshow( images[i].astype(float))
        axes[i].set_title( titles[i] )
        axes[i].axis( 'off' )
    plt.show()

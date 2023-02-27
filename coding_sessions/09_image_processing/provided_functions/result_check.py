import sys
import numpy as np
import matplotlib.pyplot as plt 

from numpy.fft import ifft2, fft2
from math import ceil
from mpl_toolkits.axes_grid1 import make_axes_locatable
import image_kernels as kernels

def padding( zero_padding, un_pad):
    """ check if the padding as well as the un padding is correctly implemented"""
    original_image = np.random.rand( 40, 50)
    pad            = (10,15)
    original_shape = np.array( original_image.shape) 
    padded_shape   = original_shape + 2* np.array( pad)
    image          = zero_padding( original_image, pad )

    if image is None:
        raise Exception( '"zero_padding" not fully implemented, please fill in the TODO flags')

    dimension_mismatch = 'Zero padding wrongly implemented, wrong dimensions returned.\nTest image has shape (40,50) and is padded with size (10,15).\nExpected shape: (60,80), got {} instead'.format(image.shape)
    try:
        if (np.array( image.shape) != padded_shape ).any(): 
            raise Exception( dimension_mismatch)
        if image[:, :15].sum() + image[:, -15:].sum() + image[:10, :].sum() + image[-10:, :].sum() != 0: 
            raise Exception( 'Zero padding wrongly implemeted, non zero values found in the padding')
        elif not np.allclose( image[10:-10, 15:-15], original_image):
            raise Exception( 'Zero padding wrongly implemeted, "inner" values of the padded image have been modified unexpectedly')
    except:
        raise Exception( dimension_mismatch)

    recovered_image = un_pad(image, pad)
    if recovered_image is None:
        raise Exception( '"un_pad" not fully implemented, please fill in the TODO flags')

    dimension_mismatch = '"un_pad" wrongly implemented, wrong dimensions returned.\nTest padded image has shape (60,80) and is un-padded with (10,15).\nExpected shape: (40,50), got {} instead'.format(image.shape)
    try:
        if (np.array( recovered_image.shape) != original_shape ).any():
            raise Exception( dimension_mismatch)
        if not np.allclose( recovered_image, original_image):
            raise Exception( 'Un-padding wrongly implemented, un-padded image does not match previous image')
    except:
        raise Exception( dimension_mismatch) 
        
    print( 'zero_padding and un_padding correctly implemented.' )



def embed_kernel( embedding_function):
    """ Check if the embed kernel is correctly implemented"""
    kernel      = kernels.disc(17, kind='linear')
    image_size  = (64,64)
    k           = np.zeros( image_size)
    k[:17, :17] = kernel 
    k           = np.roll( np.roll( k, 56, axis=0 ), 56, axis=1 ) 

    entry = embedding_function( kernel, image_size)
    if not np.allclose( k, entry):
        print( 'Bug found in embed_kernel! Showing a plot for reference' )
        fig, axes = plt.subplots( 1, 2)
        axes[0].imshow( k)
        axes[1].imshow( entry)
        axes[0].set_title( 'correct result')
        axes[1].set_title( 'your implementation')
        for ax in axes:
            ax.axis('off') 
        plt.show()
        error_msg = """ 'embed_kernel' wrongly implemented, possible reasons are:
        Wrong shifting of the kernel on the 0th index
        Rounding to the integer in the wrong 'direction'\n
        See the plot for reference"""
        raise Exception( error_msg)
    else:
        print( 'Embed_kernel correctly implemented.')
        fig, axes = plt.subplots( 1, 2, figsize=(16,7))
        axes[0].imshow( kernel)
        axes[1].imshow( entry)
        axes[0].set_title( 'original kernel')
        axes[1].set_title( 'embedded kernel')
        for ax in axes:
            ax.axis('off') 
        plt.show()



def convolution( convolution_function):
    """ Check if the passed convolution function is correclty implemented by calling it"""
    images      = list( np.load( 'data/convolution_check.npz').values() )
    image       = images[0]
    result      = images[1].astype( float)
    kernel      = kernels.pacman( 31, kind='uniform' ) #kernels.disc( 17)
    convoluted  = convolution_function( image, kernel)

    if convoluted is None:
        raise Exception( 'Please implement the {}.'.format( convolution_function.__name__) )
    desired_shape = np.array( result.shape)
    if (np.array( convoluted.shape) != desired_shape ).any():
        plot_convolutions( image, result, convoluted)
        raise Exception( 'Error in {}, shape mismatch.\nExpected {}, got {}.'.format( convolution_function.__name__, desired_shape, convoluted.shape) )
    if not np.allclose( result, convoluted, atol=5e-4): 
        plot_convolutions( image, result, convoluted)
        raise Exception( 'Error in {}, values of the convolution did not match.'.format( convolution_function.__name__)) 
    print( "{} correctly implemented.".format( convolution_function.__name__) )




def periodic_convolution( convolution_function):
    images      = list( np.load( 'data/convolution_comparison.npz').values() )
    image       = images[0].astype( float)
    result      = images[3].astype( float)
    kernel = kernels.rectangle( 41, kind='uniform' )
    convoluted  = convolution_function( image, kernel)
    if convoluted is None:
        raise Exception( 'Please implement the periodic convolution.' )
    desired_shape = np.array( result.shape)
    if (np.array( convoluted.shape) != desired_shape ).any():
        plot_convolutions( image, result, convoluted)
        raise Exception( 'Error in {}, shape mismatch\nExpected {}, got {}.'.format( convolution_function.__name__, desired_shape, convoluted.shape) )
    if not np.allclose( result, convoluted, atol=5e-4): 
        plot_convolutions( image, result, convoluted)
        raise Exception( 'Error in {}, values of the convolution did not match.'.format( convolution_function.__name__) ) 
    print( "Periodic convolution correctly implemented." )


def plot_convolutions( *images):
    """ shows some plots to compare the implementation of the convolution
    takes as many images as it has, is meant to plot 4 images"""
    print( 'function wrongly implemented, showing plot for reference' ) 
    try:
        difference = images[-1]-images[-2] 
        images = list( images)
        images.append( difference)
    except:
        pass
    titles    = [ 'image', 'desired result', 'your implementation', 'deviation to result' ]
    fig, axes = plt.subplots( 1, len( images), figsize=(13,6))
    for i in range( len(axes) ):
        gcf = axes[i].imshow( images[i] )
        divider = make_axes_locatable( axes[i]) 
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar( gcf, cax=cax )
        axes[i].axis( 'off' )
        axes[i].set_title( titles[i] ) 
    plt.show()


def segmentation( segmentation_function):
    kernel = kernels.disc( 33, kind='linear')
    kernel = segmentation_function( kernel, kernel[ 10,16])
    if kernel is None:
        raise Exception( 'please implement the segmentation')
    zeros  = kernel==0
    ones   = kernel==1
    if zeros.all() or ones.all() or kernel[11, 16]==0 or kernel[16,16] ==0 or kernel[0,0] == 1 or kernel[9,15] == 1:
        raise Exception( 'segmentation wrongly implemented, threshold or comparison has been wrongly set')
    if not (zeros+ones).all():
        raise Exception( 'segmentation wrongly implemented, values besides 0 and 1 have been returned')
    print( 'Segmentation correctly implemented.')


### Result check for the second part of the image processing 1 stuff 
def erosion_dilation( function):
    images         = np.load( 'data/erosion_dilation_check.npz')
    original_image = images['image']
    kernel         = kernels.pacman( 17, kind='ones')
    image          = function( original_image, kernel)
    try:
        result = images[ function.__name__]
    except:
        print( "non allowed function passed, it must be named 'erosion' or 'dilation', returning without check" )
        return
    zeros              = image==0
    ones               = image==1
    admissible_numbers = zeros + ones
    if not admissible_numbers.any():
        raise Exception( 'Non admissible numbers found for "{}", make sure to deploy "segmentation"'.format( function.__name__) )
    elif not np.allclose( image, result):
        plot_convolutions( original_image, result, image )
        raise Exception( '{} wrongly implemented. Possible error source is e.g. the wrong threshold for segmentation, unflipped kernel'.format( function.__name__))
    print( '{} correctly implemented'.format( function.__name__) )


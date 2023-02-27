import numpy as np
from numpy.fft import fft2, ifft2
from math import ceil, floor
import image_padding as padding


def segmentation( image, theta):
    """
    Apply binary segmentation to an image (numpy nd-array).
    Every value >= 'theta' is set to 1, the remaining image is set to 0
    Parameters:
    -----------
    image:      nd-numpy array
                image data stored as array
    theta:      float
                threshold parameter
    Returns:
    --------
    image:      nd-numpy array
                result of binary segmentation of 'image'
    """
    segmented = np.zeros(image.shape)
    threshold = 0.9999999 * theta #ommit numerical imprecision/ roundoff errors
    segmented[ image >= threshold ] = 1
    return segmented 


def embed_kernel( kernel, image_size ):
    """
    Periodically embed the kernel to the 0th index of into an image of image_size
    The embedded_kernel is supposed to be used for a convolution in fourier space
    Parameters:
    -----------
    kernel          2d-numpy array
                    frame of the kernel which
    image_size:     tuple of ints
                    desired size of the kernel, e.g. given as image.shape
    Returns:
    --------
    embedded_kernel:  2d numpy array
                      kernel embedded into an image of shape image_size
    """ 
    kernel_shift = [ -floor( x/2) for x in list(kernel.shape) ] 
    embedded_kernel = np.zeros( image_size)
    embedded_kernel[:kernel.shape[0], :kernel.shape[1]] = kernel  
    embedded_kernel = np.roll( embedded_kernel, kernel_shift, [0,1] )
    return embedded_kernel


def periodic_convolution( image, kernel):
    """
    Apply a periodic convolution of image with a kernel. It is assumed
    that the kernel is smaller than the image, hence it needs embedding.
    The Fourier transform is used for efficiency of convolution
    Parameters:
    -----------
    image:      numpy 2d-array
                image data with image.shape > kernel.shape
    kernel:     numpy 2d-array
                kernel represented as image (2d-array)
    Returns:
    --------
    convoluted: numpy 2d-array
                convolution result of image with kernel 
    """
    kernel = embed_kernel( kernel, image.shape)
    return ifft2( fft2( kernel) * fft2( image) ).real




def find_peak( image, kernel, which='min', pad_size=None): #maybe pad_size as a string option for periodic/0
    """
    Brute force of the implementation of a 2d convolution filtering
    required for grayscale opening/closing
    Parameters:
    -----------
    image:      2d-numpy array
                original image to apply the kernel on
    kernel:     2d-numpy array
                kernel for filtering should be all 0 and 1
    pad_size:   tuple of 2 ints, default None
                how much padding should be applied, defaults to kernel.shape/2
    Returns:
    --------
    image:      2d-numpy array
                image after application of the filter
    """
    if not pad_size: 
        pad_size = kernel.shape 
    image = padding.periodic_padding( image, pad_size) 
    kernel = kernel != 0 
    
    upper = [ ceil(x/2) for x in kernel.shape]
    lower = [ floor(x/2) for x in kernel.shape]
    convolved = np.zeros( image.shape)
    for i in range( upper[0], image.shape[0]-upper[0] ):
        for j in range( upper[1], image.shape[1]-upper[0] ):
            if which == 'min':
                convolved[ i,j ] = np.min(image[ i-upper[0]+1:i+lower[0]+1, j-upper[1]+1:j+lower[1]+1 ][kernel] )
            elif which == 'max':
                convolved[ i,j ] = np.max(image[ i-upper[0]+1:i+lower[0]+1, j-upper[1]+1:j+lower[1]+1 ][kernel] )
    return padding.un_padding( convolved, pad_size)


def grayscale_dilation( image, kernel):
    image = find_peak( image, kernel, 'max') #convolution of the image with the kernel
    return image


def grayscale_erosion( image, kernel ):
    image = find_peak( image, kernel, 'min') #convolution of the image with the kernel
    return image

    
def grayscale_opening( image, kernel, n=1):
    for i in range( n):
        image = grayscale_erosion( image, kernel)
    for i in range( n):
        image = grayscale_dilation( image, kernel)
    return image


def grayscale_closing( image, kernel, n=1):
    for i in range( n):
        image = grayscale_dilation( image, kernel)
    for i in range( n):
        image = grayscale_erosion( image, kernel)
    return image


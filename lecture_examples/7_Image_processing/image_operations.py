import numpy as np
from numpy.fft import fft2, ifft2
from math import ceil, floor

def edge_detection( image, kernel, threshold):
    x = np.abs( convolution( image, kernel) )
    x = segmentation( x, threshold)
    return x

def padding( image, pad, padvalue=0):
    """ Pad the image with padvalue in every direction
    The resulting image is of shape== [shape( image) + 2* pad ]
    Parameters:
    -----------
    image:      2d-numpy array
                image data stored as array
    pad:        tuple of ints
                size of pad in every direction
    padvalue:   value to be used for padding
    Returns:
    --------
    padded_image:   2d-numpy array
                    0-padded image 
    """
    if( padvalue == 0 ):
        padded_image = np.zeros( ( image.shape[0]+2*pad[0], image.shape[1]+2*pad[1] ))
    else:
        padded_image = np.full( ( image.shape[0]+2*pad[0], image.shape[1]+2*pad[1] ), padvalue )
    padded_image[ pad[0]:-pad[0], pad[1]:-pad[1] ] = image
    return padded_image

def zero_padding( image, pad):
    """
    Pad the image with 0 in every direction
    The resulting image is of shape== [shape( image) + 2* pad ]
    Parameters:
    -----------
    image:      2d-numpy array
                image data stored as array
    pad:        tuple of ints
                size of pad in every direction
    Returns:
    --------
    padded_image:   2d-numpy array
                    0-padded image 
    """
    return padding( image, pad, 0)
    

def un_pad( padded_image, pad ):
    """
    Shrink all borders of the image by the size specified in pad
    The resulting image is of shape 'shape( padded_image) - 2* pad
    Parameters:
    -----------
    padded_image:   2d-numpy array
                    image data of (previously padded) image
    pad:            tuple of ints
                    size of (previous) pad in every direction
    Returns:
    --------
    image:          2d-numpy array
                    un-padded image on all borders directions
    """
    return padded_image[ pad[0]:-pad[0], pad[1]:-pad[1] ]



def sum_convolution( image, kernel ): 
    """
    Convolute a 2d image by the kernel
    Deploys zero padding to reduce loss of information on the edges
    Parameters:
    -----------
    image:      2d-numpy array
                image data
    kernel:     2d-numpy array
                kernel/filter for convolution
    Returns:
    --------
    image    2d-numpy array
             image after application of the kernel/filter
    """
    pad_size = [ceil( n/2) for n in kernel.shape] 
    image = zero_padding( image, pad_size)
    convoluted = np.zeros( image.shape)

    lower = [ floor(n/2) for n in kernel.shape]
    upper = [ ceil(m/2)  for m in kernel.shape]
  
    for i in range( lower[0], image.shape[0]- lower[0] ):
        for j in range( lower[1], image.shape[1]- lower[1] ): 
            #convoluted[ i,j ] = np.sum( #image[ #TODO] ...
            convoluted[i,j]=np.sum(image[i-lower[0]:i+upper[0],j-lower[1]:j+upper[1]]*np.flip(kernel) )
    return un_pad( convoluted, pad_size )#TODO 


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
    kernel_shift = [ -(x//2) for x in kernel.shape ] 

    embedded_kernel = np.zeros( image_size)
    embedded_kernel[ :kernel.shape[0], :kernel.shape[1] ] = kernel
    embedded_kernel = np.roll(embedded_kernel,kernel_shift,axis=[0,1])
    return embedded_kernel



def convolution( image, kernel, onepad=False): 
    """
    Efficiently convolute a 2d image by the kernel using the FFT
    The kernel is embedded before convolution
    Parameters:
    -----------
    image:      2d-numpy array
                image data
    kernel:     2d-numpy array
                kernel/filter for convolution
    onepad:     bool
                use 1 or 0 for padding?
    Returns:
    --------
    image    2d-numpy array
             image after application of the kernel/filter
    """
    pad_size = kernel.shape
    if( onepad ):
        padded_image = padding( image, pad_size, padvalue=1 )
    else:
        padded_image = padding( image, pad_size, padvalue=0)
    kernel = embed_kernel( kernel, padded_image.shape )
    convoluted = ifft2( fft2(padded_image) * fft2(kernel) ).real
    return un_pad( convoluted, pad_size)


def periodic_convolution( image, kernel): 
    """
    Efficiently deploy periodic convolution on a 2d image by the kernel
    Parameters:
    -----------
    image:      2d-numpy array
                image data
    kernel:     2d-numpy array
                kernel/filter for convolution
    Returns:
    --------
    image    2d-numpy array
             image after application of the kernel/filter
    """
    return ifft2( fft2( image) * fft2( embed_kernel( kernel, image.shape))).real


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
    theta = 0.999999 * theta #omit numerical imprecision/ roundoff errors
    segmented[ image >= theta] = 1
    return segmented


def threshold( image, theta, limit='lower'):
    """
    element wise "clipping" of image data by the treshold 'theta'
    can use 'theta' as an upper limit -> clip every value larger than 'theta' to 'theta'
    can use 'theta' as a lower limit -> clip every value smaller than 'theta' to 'theta'
    Does clip every value upper/bigger or lower/smaller than 
    Parameters:
    -----------
    image:  numpy nd-array
            image data
    theta:  float 
            threshold parameter
    limit:  string, default 'lower'
            specify limit type of theta
    Returns:
    --------
    image   numpy nd-array
            processed image data
    """
    if limit == 'upper': 
        image = np.minimum( image, theta )
    elif limit == 'lower':
        image = np.maximum( image, theta )
    else:
        raise Exception( "illegal input for 'limit' in 'image_operations.threshold' " )
    return image


def periodic_padding(image, pad):
    """
    Periodically pad the image in every direction
    The resulting image is of shape 'shape( image) + 2* pad
    Parameters:
    -----------
    image:      2d-numpy array
                image data stored as array
    pad:        tuple of ints
                size of pad in every direction
    Returns:
    --------
    padded_image:   2d-numpy array
                    periodically-padded image 
    """
    # centered original image
    padded_image                                   = np.zeros( ( image.shape[0]+2*pad[0], image.shape[1]+2*pad[1] ) )
    padded_image[ pad[0]:-pad[0], pad[1]:-pad[1] ] = image
    # "edges" on all four sides
    padded_image[:pad[0],  pad[1]:-pad[1]] = image[-pad[0]:, :]
    padded_image[-pad[0]:, pad[1]:-pad[1]] = image[: pad[0], :]
    padded_image[pad[0]:-pad[0], -pad[1]:] = image[:, : pad[1]]
    padded_image[pad[0]:-pad[0], : pad[1]] = image[:, -pad[1]:]
    # remaining "corners"
    padded_image[ :pad[0], :pad[1]  ]  = image[ -pad[0]:, -pad[1]: ]
    padded_image[ :pad[0], -pad[1]: ]  = image[ -pad[0]:, :pad[1]  ]
    padded_image[ -pad[0]:, :pad[1]  ] = image[ :pad[0], -pad[1]: ]
    padded_image[ -pad[0]:, -pad[1]: ] = image[ :pad[0], :pad[1]  ]
    return padded_image


def erode_by_filter( image, kernel, theta =0.999, onepad=True ):
    filtered = convolution( image.astype("float"), np.fliplr(np.flipud(kernel.astype("float"))), onepad=onepad )
    return ( filtered >= theta * np.sum(kernel.flatten()) )

def dilate_by_filter( image, kernel, theta =0.9999 ):
    filtered = convolution( image.astype("float"), kernel.astype("float") )
    return ( filtered >= theta * np.max(kernel.flatten()) )

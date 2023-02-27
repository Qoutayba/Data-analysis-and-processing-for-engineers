import numpy as np
from math import ceil, floor



def zero_padding( image, pad_size):
    """
    Pad the given image with zeros, the resulting shape will be
    'image.shape + pad_size'  (using array notation)
    Parameters:
    -----------
    image:      numpy 2d-array
                image data to be padded
    pad_size:   tuple of 2 ints (or 'full')
                size of the total pad, each side will be extended with pad_size/2
    Returns:
    --------
    padded_image:   numpy 2d-array
                    padded image of size 'image.shape + pad_size' 
    """
    if isinstance( pad_size, str) and pad_size == 'full':
        pad_size = image.shape
    pad = pad_size #for easier readablity
    padded_image = np.zeros( 2*np.array( pad) + np.array( image.shape) )
    padded_image[ pad[0]:-pad[0], pad[1]:-pad[1] ] = image.copy()
    return padded_image


def constant_padding( image, pad_size):
    """
    Pad the given image constantly only in horizontal and vertical direction
    the resulting shape will be 'image.shape + pad_size'  (using array notation)
    Parameters:
    -----------
    image:      numpy 2d-array
                image data to be padded
    pad_size:   tuple of 2 ints (or 'full')
                size of the total pad, each side will be extended with pad_size/2
    Returns:
    --------
    padded_image:   numpy 2d-array
                    padded image of size 'image.shape + pad_size' 
    """
    if isinstance( pad_size, str) and pad_size == 'full':
        pad_size = image.shape
    pad = pad_size #for easier readablity
    padded_image = np.zeros( 2*np.array( pad) + np.array( image.shape) )
    padded_image[ pad[0]:-pad[0], pad[1]:-pad[1] ] = image.copy()
    padded_image[ :pad[0], pad[1]:-pad[1] ] = image[0, :]
    padded_image[ -pad[0]:, pad[1]:-pad[1] ] = image[-1, :]
    padded_image[ pad[0]:-pad[0], :pad[1] ] = np.vstack( pad[1]* [image[:, 0] ] ).T
    padded_image[ pad[0]:-pad[0], -pad[1]: ] = np.vstack( pad[1]* [image[:, -1] ] ).T
    return padded_image


def periodic_padding( image, pad_size):
    """
    Pad the given image constantly only in horizontal and vertical direction
    the resulting shape will be 'image.shape + pad_size'  (using array notation)
    Parameters:
    -----------
    image:      numpy 2d-array
                image data to be padded
    pad_size:   tuple of 2 ints (or 'full')
                size of the total pad, each side will be extended with pad_size/2
    Returns:
    --------
    padded_image:   numpy 2d-array
                    padded image of size 'image.shape + pad_size' 
    """
    if isinstance( pad_size, str) and pad_size == 'full':
        padded_image = np.hstack( 3*[image] )
        padded_image = np.vstack( 3*[padded_image] )
        return padded_image
    pad = pad_size #for easier readablity
    padded_image = np.zeros( 2*np.array( pad) + np.array( image.shape) )
    padded_image[ pad[0]:-pad[0], pad[1]:-pad[1] ] = image
    padded_image[: pad[0], pad[0]:-pad[1]] = image[-pad[0]:, :]
    padded_image[-pad[0]:, pad[0]:-pad[1]] = image[: pad[0], :]
    padded_image[pad[0]:-pad[1], -pad[1]:] = image[:, : pad[1]]
    padded_image[pad[0]:-pad[1], : pad[1]] = image[:, -pad[1]:]
    return padded_image


def un_padding( image, pad_size):
    """
    un pad the image and return the original image
    The resulting image is of shape 'image.shape - pad_size' (array notation)
    Parameters:
    -----------
    image:      numpy 2d-array
                previously padded image
    pad_size:   tuple of two ints
                size of the pad, e.g. kernel.shape
    Returns:
    --------
    image:      numpy 2d-array
                un-padded image of size 'image.shape - pad_size' 
    """
    return image[ pad_size[0]:-pad_size[0], pad_size[1]:-pad_size[1]  ]


def embed_kernel( kernel, image_shape):
    """
    Embed a kernel for convolution via Fourier transform periodiically
    centered at the 0th index.
    Parameters:
    -----------
    kernel:         numpy 2d-array
                    kernel used for convolution
    image_shape:    tuple of 2 ints
                    shape of the image, must be bigger than kernel 
    Returns:
    --------
    embedded_kernel:numpy 2d-array
                    kernel embedded that it is of shape image_shape
    """
    kernel = np.flip(kernel) 
    shift = [ -floor( x/2) for x in list(kernel.shape) ]
    embedded_kernel = np.zeros( image_shape)
    embedded_kernel[:kernel.shape[0], :kernel.shape[1]] = kernel
    embedded_kernel = np.roll( embedded_kernel, lower, axis=[0,1] )
    return embedded_kernel

    

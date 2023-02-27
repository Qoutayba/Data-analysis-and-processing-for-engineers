import numpy as np
from numpy.fft import fft2, ifft2

def do_image_stuff( image):
    """
    Compute the 2point correlation function (2pcf) of the given 2d image
    The image is assumed to be periodic and the 2pcf is centered in the corners
    Parameters:
    -----------
    image:      numpy nd-array
                periodic image data 
    Returns:
    --------
    pcf:        numpy nd-array
                2pcf of the image (periodically computed)
    """
    image = fft2(image)
    pcf = ifft2( image * np.conj( image) ).real
    return pcf/image.size


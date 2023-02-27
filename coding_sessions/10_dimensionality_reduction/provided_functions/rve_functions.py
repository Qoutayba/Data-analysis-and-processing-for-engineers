import numpy as np
from numpy.fft import fft2, ifft2

def compute_pcf( image):
    """
    Compute the 2 point correlation function of the given image
    Parameters:
    -----------
    image:      numpy 2d-array
                image data
    Returns:
    --------
    2pcf:       numpy 2d-array
                2 point correlation function of the image
    """
    if image.ndim == 1:
        flatten = True
        image = image.reshape( *2*[int(np.sqrt( len(image))) ])
    else:
        flatten = False
    c_12 = fft2( image)
    pcf = ifft2( c_12 * np.conj( c_12) ).real 
    pcf = pcf/ np.prod( image.shape) 
    if flatten is True:
        return pcf.flatten()
    else:
        return pcf

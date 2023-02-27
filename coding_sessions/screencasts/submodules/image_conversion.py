import numpy as np
from matplotlib.image import imread, imsave

def rgb_to_grayscale( image, coefficients='CCIR'):
    """
    Convert a given RGB image to grayscale by the defined conversion method
    Parameters:
    -----------
    image:          numpy nd-array
                    RGB image of shape (x,x, 3)
    coefficients:   string, default 'CCIR'
                    conversion method, implemented are
                    'CCIR'/'REC601', 'ITU-R'/'REC709', 'SMPTE'
    Returns:
    --------
    grayscale_image:numpy nd-array
                    converted image

    """
    if coefficients == 'CCIR' or coefficients == 'REC601': 
        image = 0.299*image[:,:,0] + 0.587* image[:,:,1] + 0.114*image[:,:,2]
    elif coefficients == 'ITU-R' or coefficients == 'REC709':
        image = 0.2125*image[:,:,0] + 0.7154* image[:,:,1] + 0.0721 *image[:,:,2]
    elif coefficients == 'SMPTE':
        image = 0.212*image[:,:,0] + 0.701* image[:,:,1] + 0.087 *image[:,:,2] 
    return  image 



def load_grayscale( filename, *args, **kwargs):
    """
    loads an image with matplotlib.image.imread, the given RGB image is
    immediately converted to grayscale using 'CCIR' coefficients
    color/gray values are set to unsigned integers (0-255)
    Parameters:
    -----------
    filename:   string
                full path to file
    *args, **kwargs: options directly put into matplotolib.image.imread
    Returns:
    --------
    image:      numpy 2d-array
                loaded and converted grayscale image
    """
    image = imread( filename )
    image = (255*image).astype('u1')
    return rgb_to_grayscale( image )


def load_rgb( filename, *args, **kwargs):
    """
    loads an image with matplotlib.image.imread, which returns a (x,x,3) float array
    convert the color values to unsigned integers (0-255)
    Parameters:
    -----------
    filename:   string
                full path to file
    *args, **kwargs: options directly put into matplotolib.image.imread
    Returns:
    --------
    image:      numpy 2d-array
                loaded and converted RGB image
    """
    image = imread( filename, *args, **kwargs)
    image = (255*image).astype('u1')
    return image



def float_to_u8( image):
    """
    converts any float image to a u1 scale
    assumes that the maximum amplitude in 'image' is "white" (255) and the lowest is black (0)
    """
    image = image - image.min() 
    image = (image / image.max() * 255).astype('u1')
    return image



def u8_to_float( image):
    """
    converts the given u1 image to float values
    keeps color properties, rescales the color definition on the range [0,1]
    """
    return image.astype(float) /255



def rgb_to_hsl( image):
    """
    converts an image with float rgb values [0-1] to HSL colors
    HSL -> hue, saturation, lightness
    Parameters:
    -----------
    image:  numpy nd-array
            RGB image with three color channels with values between [0,1]
    Returns:
    --------
    image:  numpy nd-array
            image converted to HSL colorscheme (also 3 channels)
    """
    hue = np.zeros( image.shape[:-1])
    saturation = np.zeros( image.shape[:-1])
    lightness = np.zeros( image.shape[:-1])

    for i in range( image.shape[0] ):
        for j in range( image.shape[0] ):
            pixel  = image[i,j,:]
            red, green, blue = pixel
            max_hue = pixel.max()
            min_hue = pixel.min() 
            if max_hue == min_hue:
                coefficient = 0
            elif max_hue == red: 
                coefficient = (green-blue)/(max_hue- min_hue)
            elif max_hue == green:
                coefficient = 2+(blue-red)/(max_hue- min_hue)
            elif max_hue == blue:
                coefficient = 4+(red-green)/(max_hue- min_hue)
            hue[i,j] = (60*coefficient) % 360
            if max_hue != 0 and min_hue != 1 :
                saturation[i,j] = (max_hue-min_hue) / (1- abs( max_hue+min_hue -1) )
            #else: #it remains 0 
            lightness[i,j] = (max_hue-min_hue) / 2 
            
    return np.array( [hue, lightness, saturation] ).T



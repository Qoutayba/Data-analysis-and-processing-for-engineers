import numpy as np
import image_operations as operate


def dilation( image, kernel):
    """
    Dilate the given image by the given kernel
    Parameters:
    -----------
    image:  2d-numpy array
            image data stored as array
    kernel  2d-numpy array
            kernel/filter used for dilation
    Returns:
    --------
    image:  2d-numpy array
            dilated input image by kernel
    """
    #image = operate.#TODO #convolution of the image with the flipped kernel
    #image = #TODO #segment the convoluted image at the correct value 
    return #TODO



def erosion( image, kernel ):
    """
    Erode the given image by the given kernel
    Parameters:
    -----------
    image:  2d-numpy array
            image data stored as array
    kernel  2d-numpy array
            kernel/filter used for erosion
    Returns:
    --------
    image:  2d-numpy array
            eroded input image by kernel
    """
    #image = #TODO #convolution of the image with the flipped kernel
    #image = #TODO #segment the convoluted image at the correct value 
    return  #TODO
    


def opening( image, kernel):
    """
    Perform opening on a given image using the specified kernel
    Opening is an erosion of the image followed by a dilation
    Parameters:
    -----------
    image:  2d-numpy array
            image data stored as array
    kernel  2d-numpy array
            kernel/filter used for opening
    Returns:
    --------
    image:  2d-numpy array
            image after performing opening
    """
    #TODO... 
    return #TODO



def closing( image, kernel):
    """
    Perform closing on a given image using the specified kernel
    Closing is an dilation of the image followed by a erosion
    Parameters:
    -----------
    image:  2d-numpy array
            image data stored as array
    kernel  2d-numpy array
            kernel/filter used for closing
    Returns:
    --------
    image:  2d-numpy array
            image after performing closing
    """
    #TODO... 
    return #TODO

#!/usr/bin/env python3
import numpy as np



def diamond(n_x, n_y=None, kind='ones'):
    """
    Return an unpadded diamond kernel of shape==(n_y,n_x)
    The diamond is basically a rotated rectangle by 45°
    The diamond is centered in the middle of the kernel image
    Parameters:
    -----------
    n_x:    int
            size in x direction (columns)
    n_y:    int, default None
            size in y direction (rows)
            If not specified it copies n_x
    kind:   string, default 'ones'
            choose between the kind of kernel:
            ones:    binary kernel with 0s and 1s
            uniform: normalized 'ones' kernel
            linear:  normalized with a gradient from outside to middle
    Returns:
    --------
    kernel: numpy 2d-array
            Kernel centered in the array of shape==( n_y,n_x)
    """
    eps = 1e-15
    if n_y is None:
        n_y = n_x
    kernel = np.zeros( [n_y,n_x])
    y      = np.linspace(-1,1,n_y)
    x      = np.linspace(-1,1,n_x)
    
    kernel = np.abs( np.outer(np.ones(n_y),x)) + np.abs(np.outer(y,np.ones(n_x)))
    if kind == 'linear' :
        kernel = np.maximum( 0, 1-kernel )
    else:
        kernel = (kernel <= 1+eps).astype(int)
        
    if kind != 'ones':
        kernel = kernel/np.sum(kernel,axis=(0,1))
    return kernel


def disc( n_x,n_y=None,kind='ones'):
    """
    Return an unpadded circular kernel of shape==(n_y,n_x)
    The circle is centered in the middle of the kernel image
    Parameters:
    -----------
    n_x:    int
            size in x direction (columns)
    n_y:    int, default None
            size in y direction (rows)
            If not specified it copies n_x
    kind:   string, default 'ones'
            choose between the kind of kernel:
            ones:    binary kernel with 0s and 1s
            uniform: normalized 'ones' kernel
            linear:  normalized with a gradient from outside to middle
            gauss:   normalized gaussian disc kernel
    Returns:
    --------
    kernel: numpy 2d-array
            Kernel centered in the array of shape==( n_y,n_x)
    """
    eps = 1e-15
    if(n_y is None):
        n_y=n_x
    y = np.outer(np.linspace(-1,1,n_y), np.ones(n_x))
    x = np.outer(np.ones(n_y), np.linspace(-1,1,n_x))
    r = x**2 + y**2
    if kind=='uniform' or kind == 'ones':
        kernel = ( r<=1+eps )
    elif kind=='linear':
        kernel = np.maximum(0, 1-r[:,:])
    elif kind=='gauss':
        kernel = np.exp( - 3*r[:,:]**2 )
    
    if kind == 'ones':
        kernel[ kernel != 0] = 1
    else:
        kernel = kernel/np.sum(kernel,axis=(0,1))
    
    return kernel

def rectangle(n_x, n_y=None, kind='ones'):
    """
    Return an unpadded rectangle kernel of shape==(n_y,n_x)
    the rectangle fills the whole kernel image
    Parameters:
    -----------
    n_x:    int
            size in x direction (columns)
    n_y:    int, default None
            size in y direction (rows)
            If not specified it copies n_x
    kind:   string, default 'ones'
            choose between the kind of kernel:
            ones:    binary kernel with 0s and 1s
            uniform: normalized 'ones' kernel
            linear:  normalized with a gradient from outside to middle
    Returns:
    --------
    kernel: numpy 2d-array
            Kernel centered in the array of shape==( n_y,n_x)
    """
    eps = 1e-15
    if (n_y is None):
        n_y=n_x 
    kernel = np.ones([n_y,n_x])
    if kind == 'ones':
        return kernel
    elif (kind != 'uniform'):
        y      = np.abs(np.linspace(-1,1,n_y))
        x      = np.abs(np.linspace(-1,1,n_x))
        kernel = np.maximum(0, 1 - np.maximum( np.outer(np.ones(n_y),x), np.outer(y,np.ones(n_x)) ) )
        
    kernel = kernel/np.sum(kernel,axis=(0,1))
    return kernel

def pacman(n, kind='ones'):
    """
    Return an unpadded pacman kernel of shape==(n,n)
    Pacman looks always to the right and 'inherits' its kind from disc
    Parameters:
    -----------
    n:      int
            size in x direction (columns)
    kind:   string, default 'ones'
            see 'disc' for reference
    Returns:
    --------
    kernel: numpy 2d-array
            Kernel centered in the array of shape==( n_y,n_x)
    """
    kernel = disc( n, kind=kind )
    for m in np.arange( -int(n/4), int(n/4) ):
        l = int(2*np.abs(m) + int(n/2) )
        kernel[m+int(n/2),l:] = 0
    if kind != 'ones':
        kernel = kernel/np.sum(kernel,axis=(0,1))
    return kernel

def laplace( kind='full'):
    """
    Return an unpadded normalized Laplacian kernel of size 3x3
    Can choose between a full Laplacian (which considers diagonals)
    or a non full Laplacian which only considers x&y direction
    Parameters:
    -----------
    kind:   string, default 'full'
            kind of the Laplacian, choose between 'full' and 'cross'
    Returns:
    --------
    kernel: numpy 2d-array
            Laplace kernel of shape 3x3
    """
    if kind == 'full':
        kernel = np.ones( ( 3,3)) * -1/8 #TODO
        kernel[1,1] = 1
    elif kind == 'cross':
        kernel = np.zeros( (3, 3) )
        kernel[1,:] = -1/4
        kernel[:,1] = -1/4
        kernel[1,1] = 1
    else: 
        raise Exception( "non defined Laplace kernel requested, choose between 'full' and 'cross' ")
    return kernel



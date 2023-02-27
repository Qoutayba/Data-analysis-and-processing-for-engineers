import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap as Cmap
import sample_sets as sample

## Plotting
def split_sets( x_one, x_two, y_hat, x_orig=None ):
    """ plot two sets of row wise aranged samples in 2d (their sets) and
    in a 3d scatterplot """ 
    n_sets = len( x_orig)//2
    colors = n_sets*['red'] + n_sets*['blue' ]
    x = np.vstack( ( x_one, x_two ) )
    if x_orig:
        fig, axes = plt.subplots( 1,4, figsize=(16,6) )
        titles = ['unsorted data', 'reference solution', '3d visualization', 'split datasets']
        xx, yy = np.meshgrid( np.arange(-1.5,1.5, 0.1), np.arange(-1.5,1.5,0.1) ) #required for 3d plot
        for i in range( len( x_orig)):
            axes[1].scatter( x_orig[i][ :,0], x_orig[i] [:,1], facecolor=colors[i], edgecolor='k')
        #3D plot
        axes[-2] = plt.subplot( 1,4,3, projection='3d')
        axes[-2].plot_surface( xx, yy, np.zeros( xx.shape), alpha=0.2, color='red' )
        axes[-2].plot( x[:,0], x[:,1], y_hat, 'o', markeredgecolor='k', markerfacecolor='gray')
        if np.random.rand() >0.5:
            axes[-2].view_init( elev=12, azim=8)
        else:
            axes[-2].view_init( elev=20, azim=10)
    else: 
        fig, axes = plt.subplots( 1, 2, figsize=(16,6) )
        titles = ['unsorted data', 'split datasets']
    axes[0].scatter( x[:,0], x[:,1], facecolor='gray', edgecolor='k')
    axes[-1].scatter( x_one[:,0], x_one[:,1], facecolor='blue', edgecolor='k' )
    axes[-1].scatter( x_two[:,0], x_two[:,1], facecolor='red', edgecolor='k' )

    for ax in axes:
        ax.set_title( titles.pop(0))
        ax.grid( ls=':')
    plt.show()



def checkerboard():
    """ Plot the default reference solution of the checkerboard """
    fig, ax =plt.subplots( figsize=(6,6) )
    ax.imshow( np.flip( sample.checkerboard_reference().T, 0) )


def sampled_checkerboard( boundaries, x_positive, x_negative, reference_checkerboard):
    """ Plot the floating point boundaries of the checkerboard task 
    also plot the samples"""
    resolution = boundaries.shape
    evaluation_matrix = np.zeros( resolution)
    evaluation_matrix[ boundaries >=0 ] = 1
    evaluation_matrix[ boundaries < 0 ] = -1
    max_eval = np.max( abs( boundaries) )
    scaling_factor = np.array( boundaries.shape )
    fig, axes = plt.subplots( 1,3, figsize=(18,6))
    x_plot = [ x_positive *scaling_factor, x_negative*scaling_factor]
    cmap = Cmap.from_list( "", [ (0., 0.6, 0.6), (0., 0.6, 0.6), (0.,0.5,0.8), "black", "yellow", (1.0, 1.0, .3), (1.0, 1.0, 0.5) ] )

    axes[0].imshow( np.flip(reference_checkerboard.T, 0) )
    axes[1].imshow( np.flip( evaluation_matrix.T, 0)  )
    density = axes[2].imshow( boundaries.T, vmin=-max_eval, vmax=max_eval, cmap=cmap )
    axes[2].scatter( x_plot[0][:,0], x_plot[0][:,1], facecolor='orange', edgecolor='k' )
    axes[2].scatter( x_plot[1][:,0], x_plot[1][:,1], facecolor='green', edgecolor='k' )

    plt.colorbar( density, ax=axes[2])
    axes[2].set_xlim( xmin=0, xmax=resolution[0] )
    axes[2].set_ylim( ymin=0, ymax=resolution[1] )
    axes[1].set_title( 'reconstructed checkerboard')
    axes[0].set_title( 'reference solution')
    axes[2].set_title( 'floating boundaries' )




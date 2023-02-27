import matplotlib.pyplot as plt
import numpy as np
import pickle
import h5py

from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import subfunctions as compute


def show_all():
    print( 'scientific vs non-scientific')
    trump_hillary()
    print( 'when to use scatterplots')
    scatter_basic() 
    print( 'how to enhance scatterplots')
    scatter_advanced() 
    print( 'from scatterplots to image plots')
    scatter_elaborate()
    print( 'multiple related lines' )
    lines()
    print( 'losses of a neural network training' )
    losses()
    print( 'plotting images' )
    imshow()
    print( 'plot with an inset' )
    insets()
    print( 'plot with a reference' )
    referencing_plot()
    plt.show()
    return





########################################
def trump_hillary():
    votes = [49, 51]
    location = [1.0,2.0]
    labels = ['Trump', 'Hillary']
    fig, axes = plt.subplots(1,2, figsize=(12,6))
    # plotting of data
    axes[0].bar( location, votes, width=0.7)
    axes[1].bar( location, votes, width=0.7)

    # style
    axes[0].set_title( 'Media plot' )
    axes[0].set_ylabel( 'Popularity voting')
    axes[0].set_ylim( 48.8, 51.2)
    axes[0].set_yticks( [49, 51] )
    axes[1].set_title( 'Scientific plot' )
    axes[1].set_ylabel( 'Population vote [%]')
    axes[1].set_ylim( ymin=0, ymax=100)
    axes[1].set_yticks( np.arange( 0, 110, 10 ) )
    axes[1].grid( ls='--', color='#AAAAAA', axis='y')
    for ax in axes:
        ax.set_xticks( location)
        ax.set_xticklabels( labels)
    return


########################################
def scatter_basic():
    scatter = np.load( 'data/scatter_data.npz') 
    n_samples = 1000
    x = scatter[ 'arr_0' ][:n_samples]
    y = scatter[ 'arr_1' ][:n_samples]

    fig, axes = plt.subplots( 1,2, figsize=(12,6) )
    ## bad/no effort plot
    axes[0].plot( np.sort( x),y)

    ### plotting of data
    bound_x, min_bound, max_bound, mean_bound = compute.compute_sample_bounds( x, y, stepsize=14 )
    #axes[1].plot( bound_x, min_bound, color='red' )
    #axes[1].plot( bound_x, max_bound, color='red', label='bounds' )
    axes[1].plot( bound_x, mean_bound, color='red', label='mean' )
    axes[1].scatter( x, y, facecolor='lightblue', edgecolor='k', label='samples')

    axes[1].set_xlim( xmin=0.2, xmax=0.8) 
    axes[1].legend()
    axes[1].set_xlabel( 'volume fraction [-]' )
    axes[1].set_ylabel( r'material property [$\frac{ \rm W}{\rm m\cdot K}$]' )
    return


########################################
def scatter_advanced():
    data = np.load( 'data/dense_scatterdata.npz')
    x = data['arr_0' ] 
    y = data['arr_1' ] *100
    fig, axes = plt.subplots( 1, 2, figsize=(12,6) )
    ## bad/no effort plot
    axes[0].scatter( x, y)

    ### better plot
    #data pre processing
    x_lin, y_lin = compute.regression( x, y, poly_order=1 )
    x_reg, y_reg = compute.regression( x, y, poly_order=2 )

    #plotting of data
    axes[1].scatter( x, y, facecolor='#00beff', edgecolor='black', label='samples')
    axes[1].plot( x_lin, y_lin, ls='--', color='#8dc63f', lw=2.5 )
    axes[1].plot( x_reg, y_reg, color='#8dc63f', lw=2.5, label='regression' )


    ## style
    axes[1].set_xlim( xmin=0, xmax=1.05*x.max() )
    axes[1].set_ylim( ymin=0, ymax=1.05*y.max() )
    axes[1].legend()
    axes[1].set_xlabel( 'model error [-]' )
    axes[1].set_ylabel( 'projection error [%]' )
    return







########################################
def scatter_elaborate():
    data = np.load( 'data/dense_scatterdata.npz')
    x = data['arr_0' ]
    y = data['arr_1' ]
    blur = data[ 'arr_2']
    fig, axes = plt.subplots( 1,2, figsize=(12,6) )
    axes[0].scatter( x, y)
    axes[1].imshow( blur)
    axes[1].axis('off' )
    # this one requires more pre-/post processing
    # it is done by a convolution of the data with a gaussian kernel.
    # The results were originally plotted with:
    # blur = np.histogram2d( x, y, bins=200)[0]
    # blur = ndimage.gaussian_filter( blur, sigma=5) 
    # plt.pcolormesh( blur.T, cmap='jet', shading='gouraud' ) 
    # the decorators were created with 'tikz', it might be possible to recreate that with plt.
    # After image_processing, you will understand how that plot was exactly created
    return



########################################
def lines():
    ### Generation of data
    x = np.arange( 0.001, 16, 0.1)
    square = lambda a=0.2, b=0, c=0.5, x=x: np.maximum( 0, a*x**2 + b*x + c )
    log = lambda a=0.2, c=0.5, x=x: np.maximum( 0, a* np.log( x+0.5) + c )
    root = lambda a=0.2, b=0.2, c=0.5, x=x: np.maximum( 0, a*x**b + c )

    set_1 = [ [0.01, 0.5, -1.5], [0.08, 0, -2.6], [0.00, 1, -3.4] ]
    set_2 = [ [1.0, 0.2], [0.8, 0.4], [1.2, -0.1] ]
    set_3 = [ [1.5, 0.5, 1.0], [2.0, 0.45, 0.8], [1.2, 0.55, 0.9] ]
    lines_square = []
    lines_log = []
    lines_root = []
    for a, b, c in set_1:
        lines_square.append( square( a, b, c) )
    for a, c in set_2:
        lines_log.append( log( a, c) )
    for a, b, c in set_3:
        lines_root.append( root( a, b, c) )
    
    ### Plot
    fig, axes = plt.subplots( 1,2, figsize=(12,4.5))
    ## bad/no effort plot
    for i in range( 3):
        axes[0].plot( x, lines_square[i])
        axes[0].plot( x, lines_log[i])
        axes[0].plot( x, lines_root[i])

    ### better plot
    colors = ['blue', 'red', 'lightblue' ]
    linestyles = [ '-', '--', '-.' ]
    ## data of plot
    label = 'setup {}: measure 1-3'
    for i in range( 3):
        if i==0:
            axes[1].plot( x, lines_square[i], color=colors[0], ls=linestyles[i], label=label.format( 1) )
            axes[1].plot( x, lines_log[i], color=colors[1], ls=linestyles[i], label=label.format( 2) )
            axes[1].plot( x, lines_root[i], color=colors[2], ls=linestyles[i], label=label.format( 3) )
        else:
            axes[1].plot( x, lines_square[i], color=colors[0], ls=linestyles[i] )
            axes[1].plot( x, lines_log[i], color=colors[1], ls=linestyles[i] )
            axes[1].plot( x, lines_root[i], color=colors[2], ls=linestyles[i] )

    ## style of plot
    axes[1].grid( ls=':', color='#AAAAAA')
    axes[1].legend()
    axes[1].set_xlabel( r'time $t$ [s]' )
    axes[1].set_ylabel( r'response $u$ [mm]' )
    axes[1].set_title( 'deformation of a rod' )
    axes[1].set_xlim( xmin=0, xmax=15.5 )
    axes[1].set_ylim( ymin=0 )
    return






########################################
def losses():
    with h5py.File('data/ANN_result.hdf5', 'r') as f:
        training_1 = f['loss_function/dset_1'][:] 
        training_2 = f['loss_function/dset_2'][:] 
    losses = np.hstack(( training_1, training_2) )
    epochs = np.arange( losses.shape[0])
    xmin, ymin = compute.get_min_vals( losses )  
    fig, axes = plt.subplots(1,2, figsize= (12,6) )

    # plot the bad/no effort plot
    axes[0].plot( epochs, training_1[:,0] )
    axes[0].plot( epochs, training_1[:,1] )
    axes[0].plot( epochs, training_2[:,0] )
    axes[0].plot( epochs, training_2[:,1] )


    # adding the data to the plot
    axes[1].semilogy( epochs, training_1[:,0], 'r',   label='Training 1', lw=2 ) 
    axes[1].semilogy( epochs, training_1[:,1], '--r', label='Validation 1', lw=2 )
    axes[1].semilogy( epochs, training_2[:,0], 'b',   label='Training 2', lw=2) 
    axes[1].semilogy( epochs, training_2[:,1], '--b', label='Validation 2', lw=2) 
    axes[1].scatter( xmin[:], ymin[:], edgecolor='k', marker='o' , facecolor='none', s=120, lw=2.5, label='minima') 

    ## Add a box in the plot which shows the 'interesting part'
    inner_ax = plt.axes( [0.61, 0.63, 0.1, 0.22] ) #plt.axes( [left, bottom, width, height ] ) #specifies the corner, as well as its size
    inner_ax.semilogy( epochs, training_1[:,0], 'r' ) 
    inner_ax.semilogy( epochs, training_1[:,1], '--r')
    inner_ax.semilogy( epochs, training_2[:,0], 'b') 
    inner_ax.semilogy( epochs, training_2[:,1], '--b') 
    inner_ax.scatter( xmin[:], ymin[:], edgecolor='k', marker='o' , facecolor='none', s=60, lw=1.5)
    inner_ax.set_xlim( 360,430)
    inner_ax.set_ylim( 0.006, 0.08)
    plt.setp(inner_ax, xticks=[] ) #remove the ticks #setp -> set_properties( ax_object, properties)
    inner_ax.tick_params( axis='y', which='both', left=False, labelleft=False)


    ## setting a nice layout on the plot
    axes[1].set_xlim( 0, 800) 
    axes[1].set_ylim( 1.5e-3, 1.5) 
    legend_style = dict( fontsize=12, facecolor=(0.8, 0.8, 0.8, 0.6), edgecolor='black') 
    compute.layout(axes[1], 'number of epochs [-]', 'loss [-]', title='loss function over training', **legend_style)
    # highlight the box which shows the 'interesting part'
    translucent_grey = ( 0.8, 0.8, 0.8, 0.35 )
    full_translucent = ( 1, 1, 1, 0)
    mark_inset( axes[1], inner_ax, loc1=3, loc2=3, facecolor=translucent_grey, edgecolor='black', lw=1.2)
    mark_inset( axes[1], inner_ax, loc1=1, loc2=3, facecolor=full_translucent, edgecolor='blue', lw=3) 
    return


########################################
def imshow():
    image_data = list( np.load( 'data/images.npz' ).values() )
    fig, axes = plt.subplots( 2,3, figsize=(12,8) )
    for i in range( 3):
        axes[0, i].imshow( image_data[i][0] )
        axes[1, i].imshow( image_data[i][1] )

    fig, axes = plt.subplots( 2,3, figsize=(12,8) )
    for i in range( 3):
        axes[0, i].imshow( image_data[i][0] )
        axes[1, i].imshow( image_data[i][1] )

    titles = ['RVE', '2PCF' ]
    for ax in axes.flatten():
        ax.axis( 'off')
    for ax in axes[:,1]:
        ax.set_title( titles.pop(0) )
    ## equivalent working syntax
    #for i in range( len(labels) ):
    #    axes[i,0].set_title( titles[i] )
    return

def insets():
    ## data creation
    c0 = 3
    c1 = 1
    c2 = 5.2
    cc= 42 
    x = np.arange(0, 26, 0.25)
    y0 = x *c0 +cc
    y1 = x**2 * c1 +cc
    y2 = x**0.5 * c2 +cc 
    ## pre processing
    intersection = 3.00
    marker = '({}, {})'.format( intersection, intersection*c0+cc )

    plt.rcParams.update( {'font.size':13} ) 
    box_color = (0.85, 0.85, 0.85, 0.3)
    legend_style = dict( fontsize=14, facecolor=(0.8, 0.8, 0.8, 0.6), edgecolor='black')

    ## plotting of data
    fig, axes = plt.subplots( 1, 2, figsize=(15,5) )
    axes[0].plot( x, y0, lw=2.5, label='{} x+{}'.format(c0, cc) )
    axes[0].plot( x, y1, lw=2.5, label='{} x**2+{}'.format(c1, cc) )
    axes[0].plot( x, y2, lw=2.5, label='{} x**0.5+{}'.format(c2, cc) ) 

    axes[1].plot( x, y0, color='blue', lw=2.5, label=r'{}$ x+{}$'.format(c0, cc) )
    axes[1].plot( x, y1, color='orange', lw=2.5, label=r'{}$ x^2+{}$'.format(c1, cc) )
    axes[1].plot( x, y2, color='red', lw=2.5, label=r'{}$ \sqrt{{x}}+{}$'.format(c2, cc) ) 
    axes[1].scatter( 100, intersection*c0+cc, fc='None', ec='k', s=100, lw=1.8, label='intersection' )

    ## adding of inset
    box = plt.axes( (0.57, 0.36, 0.16, 0.25) ) 
    box.plot( x, y1, color='orange', lw=2 )
    box.plot( x, y2, color='red', lw=2 ) 
    box.plot( x, y0, color='blue', lw=2, ls='--' )
    box.scatter( intersection, intersection*c0+cc, fc='None', ec='k', s=100, lw=1.8 ) 

    ### plot style
    axes[0].legend()
    axes[0].grid()
    mark_inset( axes[1], box, loc1=2, loc2=4, facecolor=box_color, edgecolor='k', lw=1.5 )
    axes[1].grid( ls='--', color='gray', linewidth=1.5 )
    axes[1].set_xlim( xmin=0, xmax=25)
    axes[1].set_ylim( ymin=0, ymax=700 )
    axes[1].set_title('Linear and non-linear functions')
    axes[1].set_xlabel( 'x [-]' )
    axes[1].set_ylabel( 'f(x) [-]' )
    axes[1].legend( **legend_style) 

    box.set_xlim( xmin=0.5, xmax=7)
    box.set_ylim( ymin=20, ymax=80 ) 
    box.text( 3.1, 39, marker, horizontalalignment='center', verticalalignment='center')
    box.tick_params( axis='y', which='both', left=False, labelleft=False)
    plt.setp( box, xticks=[], yticks=[] ) #setp -> set_properties
    return
    

def referencing_plot():
    ## data creation
    c0 = 3
    c1 = 1
    c2 = 5.2
    cc= 42 
    x = np.arange(0, 26, 0.25)
    y0 = x *c0 +cc
    y1 = x**2 * c1 +cc
    y2 = x**0.5 * c2 +cc 
    ## pre processing
    intersection = 3.00
    marker = '({}, {})'.format( intersection, intersection*c0+cc )
    legend_style = dict( fontsize=14, facecolor=(0.8, 0.8, 0.8, 0.6), edgecolor='black')
    box_color = (0.85, 0.85, 0.85, 0.3)
    ax_pos = [0.560, 0.61, 0.125, 0.17] #position in the figure
    inside_pos = [0.015, 0.005, -0.095, -0.14]  #position inside the axes 
    box_pos = [sum(x) for x in zip(ax_pos, inside_pos ) ] #element wise addition for lists

    fig, axes = plt.subplots( 1, 2, figsize=(16,5) )
    axes[0].plot( x, y0, color='blue', lw=2.5, label=r'{}$ x+{}$'.format(c0, cc) )
    axes[0].plot( x, y1, color='orange', lw=2.5, label=r'{}$ x^2+{}$'.format(c1, cc) )
    axes[0].plot( x, y2, color='red', lw=2.5, label=r'{}$ \sqrt{{x}}+{}$'.format(c2, cc) ) 

    box = plt.axes( (0.135, 0.13, 0.111, 0.11), facecolor=box_color) 

    axes[1].plot( x, y0, color='blue', lw=2.5, label=r'{}$ x+{}$'.format(c0, cc) )
    axes[1].plot( x, y1, color='orange', lw=2.5, label=r'{}$ x^2+{}$'.format(c1, cc) )
    axes[1].plot( x, y2, color='red', lw=2.5, label=r'{}$ \sqrt{{x}}+{}$'.format(c2, cc) ) 
    axes[1].scatter( intersection, intersection*c0+cc, fc='None', ec='k', s=100, lw=1.8, label='intersection' ) 

    ## inside plot
    inner_ax = plt.axes( ax_pos)
    inner_ax.plot( x, y0, color='blue', lw=1.5 )
    inner_ax.plot( x, y1, color='orange', lw=1.5 )
    inner_ax.plot( x, y2, color='red', lw=1.5 ) 

    ## style of plot
    axes[0].annotate( 'fig b)', fontsize=12, xy=(4.2,120), xytext=(6.5,340), 
                       arrowprops=dict(facecolor='k', shrink=0.005) ) 
    plt.setp( box, xticks=[], yticks=[] ) #setp -> set_properties
    plt.setp( box.spines.values(), lw=2.1)

    axes[0].grid( ls='--', color='gray', linewidth=1.5 )
    axes[0].set_xlim( xmin=0, xmax=25)
    axes[0].set_ylim( ymin=0, ymax=700 )
    axes[0].set_title('Figure a) Linear and non-linear functions')
    axes[0].set_xlabel( 'x [-]' )
    axes[0].set_ylabel( 'f(x) [-]' )
    axes[0].legend( **legend_style)
    axes[1].text( 3.1, 42, marker, horizontalalignment='center', verticalalignment='center')
    axes[1].set_xlim( xmin=0.5, xmax=7.5)
    axes[1].set_ylim( ymin=10, ymax=120)
    axes[1].grid( ls='--', color='gray', linewidth=1.5 )
    axes[1].legend( **legend_style, loc=4)
    axes[1].set_title('Figure b) specified part of the functions')
    axes[1].set_xlabel( 'x [-]' )
    axes[1].set_ylabel( 'f(x) [-]' )

    inner_ax.set_xlim( xmin=0, xmax=25)
    inner_ax.set_ylim( ymin=0, ymax=700 )
    inner_ax.set_title('fig a)' )
    inner_ax.tick_params( axis='y', which='both', left=False, labelleft=False)
    plt.setp( inner_ax, xticks=[], yticks=[] )

    inner_box = plt.axes( box_pos, facecolor=box_color )
    inner_box.tick_params( axis='y', which='both', left=False, labelleft=False)
    plt.setp( inner_box, xticks=[], yticks=[] )
    plt.setp( inner_ax.spines.values(), lw=2.1)
    inner_box.set_title('b)')
    return


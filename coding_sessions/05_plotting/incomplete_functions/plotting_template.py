def default_style( ax,x, *args, **kwargs):
    """
    Write your documentation here
    """
    import matplotlib.pyplot as plt
    ax.set_xlim( x.min(), x.max() )
    ax.grid(ls='--', color='#AAAAAA',linewidth=1.25 )
    ax.legend(facecolor='red', edgecolor='blue')
    ax.set_title('my functions',fontsize='x-large')
    return


def labels( ax, xlabel, ylabel ):
    """""
    Write your documentation here
    """""
    ax.set_xlabel( xlabel,fontsize='x-large')
    ax.set_ylabel( ylabel,fontsize='x-large')
    return

def default_style1( ax, *args, **kwargs):
    """
    Write your documentation here
    """
    
    ax.grid(ls='--', color='#AAAAAA',linewidth=1.25 )
    ax.legend(facecolor='gray', edgecolor='blue')
    ax.set_title('my functions',fontsize='x-large')
    return


def labels1( ax, xlabel, ylabel ):
    """""
    Write your documentation here
    """""
    ax.set_xlabel( xlabel,fontsize='x-large')
    ax.set_ylabel( ylabel,fontsize='x-large')
    return
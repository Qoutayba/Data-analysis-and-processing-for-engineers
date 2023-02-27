import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


## Data creation
x = np.arange( 0, 10, 0.1 )
polynomial = lambda a, b, x: a*x**2 + b*x
root = lambda c, x: c* x**0.5

### slider functions
## These functions work only inside the script because they access objects global to this object 
## (axes, x, a, b, c)
def plot_poly( slider_value):    
    axes[0].lines[0].remove() #clear previous line
    axes[0].plot( x, polynomial( a.val, b.val, x), color='blue' )

def plot_root( slide_value):
    axes[1].clear() #clear previous line
    axes[1].plot( x, root( slide_value, x), color='blue' )

##plot creation
fig, axes = plt.subplots( 1, 2, figsize=( 9, 5) )
axes[0].plot( x, polynomial( 1, 0, x) )
axes[1].plot( x, root( 1, x) )

### optional other plot decorators
axes[0].set_ylim(ymax=50)
axes[1].set_ylim(ymax=10)


### slider objects
# add_axes( [ x_location, y_location, width, height ]) 
a_pos = fig.add_axes( [0.1, 0.95, 0.3, 0.04] ) #upper left slider
b_pos = fig.add_axes( [0.1, 0.90, 0.3, 0.04] ) #lower left slider
c_pos = fig.add_axes( [0.6, 0.95, 0.3, 0.04] ) #right slider

a = Slider( a_pos, valmin=0, valmax=5, valinit=1, valstep=0.2, label=r'$*\cdot x^2$')
b = Slider( b_pos, valmin=-10, valmax=10, valinit=0, valstep=0.2, label=r'$*\cdot x$')

c = Slider( c_pos, valmin=0, valmax=5, valinit=1, valstep=0.2, label=r'$*\cdot \sqrt{x}$')

a.on_changed( plot_poly)
b.on_changed( plot_poly)
c.on_changed( plot_root)


plt.show()

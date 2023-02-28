import numpy as np
import matplotlib.pyplot as plt
###
### Data loading and array allocation
## image is given as RGB -> red green blue channels
uni = plt.imread( 'data/uni_stuttgart.png')[...,:3] 
no_sky = uni.copy()
only_sky = uni.copy()

### Find the sky and extract it from the original image
## masks are arrays which denote positions of 'True' conditions
## create a mask of the sky (pixel values where there should be sky
## sky has high blue  values and low other values
sky_mask = (uni[...,0] < 0.4) * (uni[...,1] < 0.4)* (uni[...,2] > 0.5) 
## * here serves as logical and
## create copies of the original image an only keep values which are sky/not sky
no_sky[sky_mask] = 0 
only_sky[~sky_mask] = 0

### Plots
fig, axes = plt.subplots(1,3)
axes[0].imshow( uni)
axes[1].imshow( no_sky) 
axes[2].imshow( only_sky) 
axes[0].set_title( 'original image' )
axes[1].set_title( 'no sky' )
axes[2].set_title( 'only sky' )
for ax in axes:
    ax.axis('off')
plt.show()

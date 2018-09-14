# calculate mapping table and corrected line via forward correction
# corresponding literature: W Cai et al (2014) 
from tifffile import imread
import numpy as np
import math

def forward_correction(filename):
    x = imread(filename)
    axes = 'YX'

    N = x.shape[1] # no. of pixels in each scan line (along x axis)
    n1 = list(range(0, N)) # indices of distorted (nonlinear) line
    n2 = np.zeros(N) # mapping table between pixel no. in nonlinear and linear scan
    i = [] # indices of corrected (linear) line
    
    for index in range(0, N):    
        n2_float = ((N-1)/2)*(1-math.cos((math.pi*n1[index]/(N-1))))# calculation of the mapping table
        n2_int = int(round(n2_float)) # rounding of k_float to get integers
        n2[index] = n2_int
        if n2_int not in i:
            i.append(n2_int)
    
# create array for corrected image with same shape as distorted image
    y = np.zeros(x.shape)
    
# fill in corresponding pixel values of x into corrected position y
# edges of image: redundant data is averaged
# center of image: undersampled data, leading to gaps (black lines in corrected image)
    for y_pos in range(0, x.shape[0]):
        for idx_i in range(0, len(i)):
            x_pos_dist = [m for m, e in enumerate(n2) if e == i[idx_i]]
            avg = sum(x[y_pos, x_pos] for x_pos in x_pos_dist)/len(x_pos_dist)
            y[y_pos,i[idx_i]] = avg
    
    return x, y
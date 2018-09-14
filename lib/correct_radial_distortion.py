#################################################################################
# for radial distortion correction use as in example below:
#  
# for the hybrid method:
# distorted, corrected = correct_radial_distortion.hybrid_method(filename)
#
#
# or for the more time efficient inverse method:
# distorted, corrected = correct_radial_distortion.inverse_method(filename)
#       
# filename must be of a 2 dimensional image or 3 dimensional image stack
#################################################################################
# to show and compare input and output use:
#
# for 2D image:
# correct_radial_distortion.show_images(distorted, corrected)
#
# for frame of a 3D stack:
# correct_radial_distortion.show_frames(distorted, corrected, frame)
#################################################################################

from tifffile import imread
import numpy as np
import os
import math
import matplotlib.pyplot as plt

# calculate mapping table and corrected line via forward correction
# corresponding literature: W Cai et al (2014) 
def mapping_forward(x):
   
    N = x.shape[1] # no. of pixels in each scan line (along x axis)
    n1 = list(range(0, N)) # indices of distorted (nonlinear) line
    n2 = np.zeros(N) # mapping table between pixel no. in nonlinear and linear scan
    i = [] # indices of corrected (linear) line
    
    for index in range(0, N):    
        n2_float = ((N-1)/2)*(1-math.cos((math.pi*n1[index]/(N-1))))# calculation of the mapping table
        n2_int = int(round(n2_float)) # rounding of k_float to get integers
        n2[index] = n2_float
        if n2_int not in i:
            i.append(n2_int)
            
    return n2, i

def mapping_inverse(x):
        
    N = x.shape[1]         
    n1 = np.zeros(N) # indices of distorted (nonlinear) line
    n2 = list(range(0, N))

# use coordinates of the pixels in the corrected image to calculate and 
# find the corresponding pixels in the original distorted image
    for index in n2:
        n1_float = ((N-1)/math.pi)*math.acos(1-(2*n2[index])/(N-1))
        n1_int = int(round(n1_float))
        n1[index] = n1_float
    return n1

# uniform scanning speed equals sinusoidal scanning speed at both demarcation points
# in between the demarcation points v_sin > v_uniform --> undersampling --> use inverse method
# for t < tL or t > tR: v_sin < v_uniform --> oversampling, redundant data --> use forward method
# n_right, n_left are the corresponding sample points (uniform scan) at tR, tL respectively
# literature: L Xu et al (2011), Fig. 3, equation (12), (13)
# calculate right demarcation point (right cut off for inverse method)
def demarcation_points(x):
    edges_proportion = 0.15
    n_left = edges_proportion * x.shape[1]
    n_right = (1-edges_proportion) * x.shape[1]
    return n_left, n_right

def pad_image(x):
    x_pad = np.zeros((400, 384))
# pad image with zeros to get full sinusoidal scanning range
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            x_pad[i][j+42] = x[i][j]
    return x_pad

def open_image(filename):
    
    curdir = os.getcwd()
    x = imread('%s' % filename)
    axes = 'ZYX'
    return x

def crop_and_zoom_image(img, target_x_size, tol=0):
    mask = img>tol
    y = img[np.ix_(mask.any(1), mask.any(0))]
    
    from scipy.ndimage import zoom
    zoom_factor_xaxis = target_x_size/y.shape[1]
    y_zoom = zoom(y, (1, zoom_factor_xaxis))
    return y_zoom


def hybrid_method(filename):
    
    stack_orig = open_image(filename)
    y_list = []
    
    dim = len(stack_orig.shape)
    if dim == 3:
        for z_pos in range(0, stack_orig.shape[0]):
            x_orig = stack_orig[z_pos]
            x = pad_image(x_orig)

            n2, i = mapping_forward(x)
            n1 = mapping_inverse(x)
            n_left, n_right = demarcation_points(x)

            N = x.shape[1]
            m_tick_list = []

        # m_tick(x_pos) = no. of sinusoidally scanned points to which the uniformly scanned point
        # at position x_pos corresponds
        # m_tick_list = list of m_ticks over all x_pos of the corrected image
            for x_pos in range(0, x.shape[1]):
                m_tick = [counter for counter, value in enumerate(n2) if int(round(value)) == x_pos]
                m_tick_list.append(m_tick)

        # create array for corrected image with same shape as distorted image
            epsilon = 0.0001
            y = np.zeros(x.shape)
            for y_pos in range(0, y.shape[0]):
                for x_pos in range(0, y.shape[1]):
                    if 0 <= x_pos <= n_left or n_right <= x_pos <= N:
                        pixelvalue = sum((1/((abs(n2[k]-x_pos))+epsilon)) * x[y_pos, k] for k in m_tick_list[x_pos])/sum((1/((abs(n2[k]-x_pos))+epsilon) for k in m_tick_list[x_pos]))    
                        y[y_pos, x_pos] = pixelvalue
                    else:
                        n1_x = n1[x_pos]            
                        nn_below = math.floor(n1_x)
                        nn_above = math.ceil(n1_x)
                        d_below = n1_x - nn_below
                        d_above = nn_above - n1_x 
                        pixelvalue = d_below * x[y_pos, nn_below] + d_above * x[y_pos, nn_above]
                        y[y_pos, x_pos] = pixelvalue
            
            y_crop_zoom_2d = crop_and_zoom_image(y, x_orig.shape[1])
            
            y_list.append(y_crop_zoom_2d)
        y_stack = np.stack(y_list)
        return stack_orig, y_stack

    elif dim == 2:
        x_orig = stack_orig
        x = pad_image(x_orig)
        n2, i = mapping_forward(x)
        n1 = mapping_inverse(x)
        n_left, n_right = demarcation_points(x)
        N = x.shape[1]
        m_tick_list = []
    # m_tick(x_pos) = no. of sinusoidally scanned points to which the uniformly scanned point
    # at position x_pos corresponds
    # m_tick_list = list of m_ticks over all x_pos of the corrected image
        for x_pos in range(0, x.shape[1]):
            m_tick = [counter for counter, value in enumerate(n2) if int(round(value)) == x_pos]
            m_tick_list.append(m_tick)
    # create array for corrected image with same shape as distorted image
        epsilon = 0.0001
        y = np.zeros(x.shape)
        for y_pos in range(0, y.shape[0]):
            for x_pos in range(0, y.shape[1]):
                if 0 <= x_pos <= n_left or n_right <= x_pos <= N:
                    pixelvalue = sum((1/((abs(n2[k]-x_pos))+epsilon)) * x[y_pos, k] for k in m_tick_list[x_pos])/sum((1/((abs(n2[k]-x_pos))+epsilon) for k in m_tick_list[x_pos]))  
                    y[y_pos, x_pos] = pixelvalue
                else:
                    n1_x = n1[x_pos]            
                    nn_below = math.floor(n1_x)
                    nn_above = math.ceil(n1_x)
                    d_below = n1_x - nn_below
                    d_above = nn_above - n1_x 
                    pixelvalue = d_below * x[y_pos, nn_below] + d_above * x[y_pos, nn_above]
                    y[y_pos, x_pos] = pixelvalue
        
        y_crop_zoom_2d = crop_and_zoom_image(y, x_orig.shape[1])
        
        return x_orig, y_crop_zoom_2d

    else:
        print('file must be a 2 dimensional image or a 3 dimensional image stack')


def inverse_method(filename):
    
    stack_orig = open_image(filename)
    y_list = []
    
    dim = len(stack_orig.shape)
    if dim == 3:
        for z_pos in range(0, stack_orig.shape[0]):
            x_orig = stack_orig[z_pos]
            x = pad_image(x_orig)

            n2, i = mapping_forward(x)
            n1 = mapping_inverse(x)

            N = x.shape[1]
            m_tick_list = []

        # m_tick(x_pos) = no. of sinusoidally scanned points to which the uniformly scanned point
        # at position x_pos corresponds
        # m_tick_list = list of m_ticks over all x_pos of the corrected image
            for x_pos in range(0, x.shape[1]):
                m_tick = [counter for counter, value in enumerate(n2) if int(round(value)) == x_pos]
                m_tick_list.append(m_tick)

        # create array for corrected image with same shape as distorted image
            epsilon = 0.0001
            y = np.zeros(x.shape)
            for y_pos in range(0, y.shape[0]):
                for x_pos in range(0, y.shape[1]):
                    n1_x = n1[x_pos]            
                    nn_below = math.floor(n1_x)
                    nn_above = math.ceil(n1_x)
                    d_below = n1_x - nn_below
                    d_above = nn_above - n1_x 
                    pixelvalue = d_below * x[y_pos, nn_below] + d_above * x[y_pos, nn_above]
                    y[y_pos, x_pos] = pixelvalue
            
            y_crop_zoom_2d = crop_and_zoom_image(y, x_orig.shape[1])
            
            y_list.append(y_crop_zoom_2d)
        y_stack = np.stack(y_list)
        return stack_orig, y_stack

    elif dim == 2:
        x_orig = stack_orig
        x = pad_image(x_orig)
        n2, i = mapping_forward(x)
        n1 = mapping_inverse(x)
        N = x.shape[1]
        m_tick_list = []
    # m_tick(x_pos) = no. of sinusoidally scanned points to which the uniformly scanned point
    # at position x_pos corresponds
    # m_tick_list = list of m_ticks over all x_pos of the corrected image
        for x_pos in range(0, x.shape[1]):
            m_tick = [counter for counter, value in enumerate(n2) if int(round(value)) == x_pos]
            m_tick_list.append(m_tick)
    # create array for corrected image with same shape as distorted image
        epsilon = 0.0001
        y = np.zeros(x.shape)
        for y_pos in range(0, y.shape[0]):
            for x_pos in range(0, y.shape[1]):
                n1_x = n1[x_pos]            
                nn_below = math.floor(n1_x)
                nn_above = math.ceil(n1_x)
                d_below = n1_x - nn_below
                d_above = nn_above - n1_x 
                pixelvalue = d_below * x[y_pos, nn_below] + d_above * x[y_pos, nn_above]
                y[y_pos, x_pos] = pixelvalue
        
        y_crop_zoom_2d = crop_and_zoom_image(y, x_orig.shape[1])
        
        return x_orig, y_crop_zoom_2d

    else:
        print('file must be a 2 dimensional image or a 3 dimensional image stack')




def show_frames(distorted, corrected, frame):
    import matplotlib.pyplot as plt
    dist = distorted
    corr = corrected
    frame = frame
    fontsize = 20
    f, axarr = plt.subplots(1, 2)
    height = 10
    width = 10
    f.set_figwidth(height)
    f.set_figheight(width)

    plt.subplots_adjust(left = 0.01, right = 0.99, wspace = 0.3, hspace = 0.1)

    axarr[0].imshow(dist[frame], cmap='magma')
    axarr[0].set_title('distorted image', fontsize=fontsize, y=1.08)
    axarr[1].imshow(corr[frame], cmap='magma')
    axarr[1].set_title('hybrid correction', fontsize=fontsize, y=1.08)

    None
    
def show_images(distorted, corrected):
    import matplotlib.pyplot as plt
    dist = distorted
    corr = corrected
    fontsize = 20
    f, axarr = plt.subplots(1, 2)
    height = 10
    width = 10
    f.set_figwidth(height)
    f.set_figheight(width)

    plt.subplots_adjust(left = 0.01, right = 0.99, wspace = 0.3, hspace = 0.1)

    axarr[0].imshow(dist, cmap='magma')
    axarr[0].set_title('distorted image', fontsize=fontsize, y=1.08)
    axarr[1].imshow(corr, cmap='magma')
    axarr[1].set_title('hybrid correction', fontsize=fontsize, y=1.08)

    None
from tifffile import imread
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy import interpolate

class Correction:
    """Modules to correct radial distortion of sinusoidally scanned images.

    input : filename (tif) of a distorted 2D image or 3D volume
    output : input image/volume and corrected image/volume as numpy arrays

    you can choose between hyrbid_method (uses forward correction for the edges and 
    inverse_correction for the middle part) or inverse_method (uses inverse_correction only)

    spline interpolation is used during the inverse_correction which makes correction process
    very slow

    Example
    -------
    correction_imgX = Correction('path_to/X_name.tif')

    input_X, output_X = correction_imgX.hybrid_method()

    or:

    input_X, output_X = correction_imgX.inverse_method()
    
    -------

    visualization:
    show_images = correction_imgX.show_images
    show_frames = correction_imgX.show_frames

    show_images(input_X, output_X)
    show_frames(input_X, output_X, 10)
    """


    def __init__(self, filename):
        self.stack_orig = self.open_image(filename)
        # check dimension of input, if 2, add dummy dimension
        self.input_dim = len(self.stack_orig.shape)
        if self.input_dim == 2:
            self.stack_orig = self.stack_orig[np.newaxis, :, :]


    def mapping_forward(self, x):
        # calculate mapping table and corrected line via forward correction
        # corresponding literature: W Cai et al (2014) 
        N = x.shape[1] # no. of pixels in each scan line (along x axis)
        n1 = list(range(0, N)) # indices of distorted (nonlinear) line
        n2 = np.zeros(N) # mapping table between pixel no. in nonlinear and linear scan
        i = []
        for index in range(0, N):    
            n2_float = ((N-1)/2)*(1-math.cos((math.pi*n1[index]/(N-1))))# calculation of the mapping table
            n2_int = int(round(n2_float)) # rounding of k_float to get integers
            n2[index] = n2_float
            if n2_int not in i:
                i.append(n2_int)    
        return n2

    def mapping_inverse(self, x):
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
    
    def forward_correction(self, x, x_pos, y_pos, n2, m_tick_list):        
        epsilon = 0.0001
        pixelvalue = sum((1/((abs(n2[k]-x_pos))+epsilon)) * x[y_pos, k] for k in m_tick_list[x_pos])/sum((1/((abs(n2[k]-x_pos))+epsilon) for k in m_tick_list[x_pos]))
        return pixelvalue


    def inverse_correction(self, x, x_pos, y_pos, n1, y):
        n1_x = n1[x_pos]                
        nn_below = math.floor(n1_x)
        nn_above = math.ceil(n1_x)
        if nn_above+1 < y.shape[1] and nn_below-1 > 0 and y_pos-1 > 0 and y_pos+1 < y.shape[0]:
            x_adjacent = [nn_below, nn_above, nn_below, nn_above, nn_below, nn_above]
            y_adjacent = [y_pos, y_pos, y_pos+1, y_pos+1, y_pos-1, y_pos-1]


            values = [x[y_pos, nn_below], x[y_pos, nn_above], x[y_pos+1, nn_below], x[y_pos+1, nn_above], x[y_pos-1, nn_below], x[y_pos-1, nn_above]]
                    
            tck = interpolate.bisplrep(x_adjacent, y_adjacent, values, kx=1, ky=1)
            pixelvalue = interpolate.bisplev(n1_x, y_pos, tck)
                                        
        else:
                    pixelvalue = (x[y_pos, nn_below] + x[y_pos, nn_above]) / 2
        return pixelvalue


    def demarcation_points(self, x):
        # uniform scanning speed equals sinusoidal scanning speed at both demarcation points
        # in between the demarcation points v_sin > v_uniform --> undersampling --> use inverse method
        # for t < tL or t > tR: v_sin < v_uniform --> oversampling, redundant data --> use forward method
        # n_right, n_left are the corresponding sample points (uniform scan) at tR, tL respectively
        # literature: L Xu et al (2011), Fig. 3, equation (12), (13)
        # calculate right demarcation point (right cut off for inverse method)
        edges_proportion = 0.15
        n_left = edges_proportion * x.shape[1]
        n_right = (1-edges_proportion) * x.shape[1]
        return n_left, n_right

    def pad_image(self, x):
        x_pad = np.zeros((400, 384))
        # pad image with zeros to get full sinusoidal scanning range
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                x_pad[i][j+42] = x[i][j]
        return x_pad

    def open_image(self, filename):
        curdir = os.getcwd()
        x = imread('%s' % filename)
        axes = 'ZYX'
        return x

    def crop_and_zoom_image(self, img, target_x_size, tol=0):
        mask = img>tol
        y = img[np.ix_(mask.any(1), mask.any(0))]
        from scipy.ndimage import zoom
        zoom_factor_xaxis = target_x_size/y.shape[1]
        y_zoom = zoom(y, (1, zoom_factor_xaxis))
        return y_zoom


    def hybrid_method(self):
        y_list = []
        stack_orig = self.stack_orig
        for z_pos in range(0, stack_orig.shape[0]):
            x_orig = stack_orig[z_pos]
            x = self.pad_image(x_orig)
            N = x.shape[1]
            n1 = self.mapping_inverse(x)
            n2 = self.mapping_forward(x)
            n_left, n_right = self.demarcation_points(x)
            # m_tick(x_pos) = no. of sinusoidally scanned points to which the uniformly scanned point
            # at position x_pos corresponds
            # m_tick_list = list of m_ticks over all x_pos of the corrected image
            m_tick_list = []
            for x_pos in range(0, x.shape[1]):
                m_tick = [counter for counter, value in enumerate(n2) if int(round(value)) == x_pos]
                m_tick_list.append(m_tick)
            
            y = np.zeros(x.shape)
            for y_pos in range(0, y.shape[0]):
                for x_pos in range(0, y.shape[1]):
                    if 0 <= x_pos <= n_left or n_right <= x_pos <= N:
                        y[y_pos, x_pos] = self.forward_correction(x, x_pos, y_pos, n2, m_tick_list)
                    else:
                        y[y_pos, x_pos] = self.inverse_correction(x, x_pos, y_pos, n1, y)
            y_crop_zoom_2d = self.crop_and_zoom_image(y, x_orig.shape[1])
            y_list.append(y_crop_zoom_2d)
        y_stack = np.stack(y_list)
    
        if self.input_dim == 2:
            return stack_orig[0], y_stack[0]
        else:
            return stack_orig, y_stack
        
    def inverse_method(self):
        stack_orig = self.stack_orig
        y_list = []
        for z_pos in range(0, stack_orig.shape[0]):
            x_orig = stack_orig[z_pos]
            x = self.pad_image(x_orig)
            N = x.shape[1]
            n1 = self.mapping_inverse(x)
            y = np.zeros(x.shape)
            for y_pos in range(0, y.shape[0]):
                for x_pos in range(0, y.shape[1]):
                    y[y_pos, x_pos] = self.inverse_correction(x, x_pos, y_pos, n1, y)
            y_crop_zoom_2d = self.crop_and_zoom_image(y, x_orig.shape[1])
            y_list.append(y_crop_zoom_2d)
        y_stack = np.stack(y_list)
    
        if self.input_dim == 2:
            return stack_orig[0], y_stack[0]
        else:
            return stack_orig, y_stack

    def show_frames(self, distorted, corrected, frame):
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
        axarr[1].set_title('correction', fontsize=fontsize, y=1.08)
        None
    
    def show_images(self, distorted, corrected):
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
        axarr[1].set_title('correction', fontsize=fontsize, y=1.08)
        None
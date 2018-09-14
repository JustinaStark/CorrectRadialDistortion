from skimage.measure import compare_mse, compare_psnr, compare_ssim
from csbdeep.utils import normalize
import numpy as np
import os

# Define function to normalize image x (restored image) and y (ground truth) 
def norm_minmse(y, x):
    y = normalize(y,0.1,99.9)
    x = x - x.mean()
    y = y - y.mean()
    scale = np.cov(x.flatten(),y.flatten())[0,1]/np.var(x.flatten())
    x = scale*x
    return y,x

def calculate_metric_values(y, x):

	y_norm, x_norm = norm_minmse(y, x)

	mse = compare_mse(y_norm, x_norm)
	psnr = compare_psnr(y_norm, x_norm, data_range=1)
	ssim = compare_ssim(y_norm, x_norm, data_range=1)

	print('mse  = ', mse)
	print('psnr =', psnr)
	print('ssim = ', ssim)


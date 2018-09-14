import os, errno
import glob
import numpy as np
from tifffile import imread, imsave
from matplotlib.pyplot import imshow

def list_files(directory, fileformat):
    return(f for f in os.listdir(directory) if f.endswith(fileformat))

def replace_in_filenames(directory, fileformat, target, replacement):
    gen_filenames = list_files(directory, fileformat)
    for filename in gen_filenames:
        os.rename(os.path.join(directory, filename), os.path.join(directory, filename.replace('%s' % target, '%s' % replacement)))

def get_z_pos(directory, fileformat, output_name='z_pos.txt'):
    gen_filenames = list_files(directory, fileformat)
    list_filenames = []
    z_pos = []
    for f in gen_filenames:
        list_filenames.append(f)

    for f in range(0, len(list_filenames)):
        temp = list_filenames[f][:7]
        if temp not in z_pos:
            z_pos.append(temp)

    

    textfile = open('%s' %  output_name, 'w')

    for item in z_pos:
        textfile.write('%s\n' % item)


def create_3d_volume(directory, axes, z_pos_file, fileformat):
    z_pos = open('%s/%s' % (directory, z_pos_file)).read().splitlines()
    try:
    	os.makedirs('volumes')
    except OSError as e:
    	if e.errno != errno.EEXIST:
    		raise

    for k in range(1, 46):
        images = []
        for z in z_pos:
            temp = imread('%s/%s_%s.%s' % (directory, z, k, fileformat))
            images.append(temp)
        vol = np.stack(images)
        imsave('%s/%s/vol_%s.%s' % (directory, 'volumes', k, fileformat), vol)
    

def average_img_with_different_noise(directory, z_pos_file, fileformat):
    z_pos = open('%s/%s' % (directory, z_pos_file)).read().splitlines()
    axes = 'XY'
    try:
    	os.makedirs('averages')
    except OSError as e:
    	if e.errno != errno.EEXIST:
    		raise

    for i in range(0, len(z_pos)):
        imglist = glob.glob('%s/%s*.%s' % (directory, z_pos[i], fileformat))

        first = True
        for img in imglist:
            temp = imread(img)
            temp = temp.astype('uint32')
            if first:
                sumimg = temp
                first = False
            else:
                sumimg = sumimg + temp
        avgarr = sumimg/len(imglist)
        imsave('%s/%s/avg_%s.%s' % (directory, 'averages', z_pos[i], fileformat), avgarr)        

def main():
    directory = os.getcwd()
    fileformat = 'tif'
    z_pos_file = 'z_pos.txt'
    axes = 'ZYX'

    to_be_deleted_from_filename = ['(',')']
    for d in to_be_deleted_from_filename:
        replace_in_filenames(directory, fileformat, '%s' % d, '')
    replace_in_filenames(directory, fileformat, ' ', '_')
    replace_in_filenames(directory, fileformat, 'Liver_940nm_170mW__x=37.950_y=29.150_z=16.', 'z16')
    get_z_pos(directory, fileformat)
    create_3d_volume(directory, axes, z_pos_file, fileformat)
    average_img_with_different_noise(directory, z_pos_file, fileformat)


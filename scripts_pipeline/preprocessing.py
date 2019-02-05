import nrrd 
import numpy as np 
from skimage.filters import threshold_triangle
from skimage.measure import label, regionprops
import time


''' 
Fisrt step is background subtraction using Fiji subtract background with settings:
Rolling ball radius 20 pixels
Disable smoothing
'''

def segment_triangle(img, remove=True):
    '''
    Image segmentation using triangle thresholding algorithm to get all blobs
    If remove: exclude blobs with less than 250 voxels
    '''
    print("Segmentation...")
    start = time.time()

    # thresholding
    img[img==0] = 10
    thres = threshold_triangle(img)
    img_thresholded = np.zeros(img.shape, dtype=img.dtype)
    img_thresholded[img>thres] = 255

    # exclude blobs with less than 250 voxels
    if remove:
        labeled_img = label(img_thresholded, neighbors=8)
        region_prop = regionprops(labeled_img)
        for i in range(len(region_prop)):
            if region_prop[i].area <= 250:
                img_thresholded[labeled_img==region_prop[i].label] = 0
    
    end = time.time()
    print("...Running time is {}".format(end-start))
    
    return img_thresholded


def find_local_max(img, mask=None):
    '''
    Find local maxima using 3D version of maximum filter
    '''
    print("Find local maxima...")
    start = time.time()

    img_max = np.zeros(img.shape, dtype=img.dtype)
    min_distance = 2
    threshold_abs = 150
    for i in range(min_distance, img.shape[0]-min_distance):
        for j in range(min_distance, img.shape[1]-min_distance):
            for k in range(min_distance, img.shape[2]-min_distance):
                img_ijk = img[i-min_distance:i+min_distance+1, j-min_distance:j+min_distance+1, k-min_distance:k+min_distance+1]

                if img[i,j,k] == img_ijk.max() and img[i,j,k] > threshold_abs:
                    img_max[i,j,k] = img[i,j,k]
    
    if mask is not None:
        mask[mask!=0] = 1
        img_max = img_max * mask

    end = time.time()
    print("...Running time is {}".format(end-start))

    return img_max


def main():

    file_path = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/test2/L2_20180504_neuron0/'
    img_name = 'BGsubtract_prod_mask_0.nrrd'

    img, head = nrrd.read(file_path+img_name)

    # segmentation
    seg_img = segment_triangle(img)
    nrrd.write(file_path+'seg_'+img_name, seg_img)

    # find local maxima
    img_max = find_local_max(img, seg_img)
    nrrd.write(file_path+'max_'+img_name, img_max)


if __name__ == "__main__":
    main()
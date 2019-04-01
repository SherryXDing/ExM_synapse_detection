from vigra.filters import multiBinaryClosing
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import minimum_filter
from skimage.morphology import h_minima, reconstruction, square, disk, cube
from skimage import io
import numpy as np
import math
from vigra.analysis import watershedsNew


def tif_read(file_name):
    """
    read tif image in (rows,cols,slices) shape
    """
    im = io.imread(file_name)
    im_array = np.zeros((im.shape[1],im.shape[2],im.shape[0]), dtype=im.dtype)
    for i in range(im.shape[0]):
        im_array[:,:,i] = im[i]
    return im_array


def tif_write(im_array, file_name):
    """
    write an array with (rows,cols,slices) shape into a tif image
    """
    im = np.zeros((im_array.shape[2],im_array.shape[0],im_array.shape[1]), dtype=im_array.dtype)
    for i in range(im_array.shape[2]):
        im[i] = im_array[:,:,i]
    io.imsave(file_name,im)
    return None


def imimposemin(I, BW, conn=None, max_value=255):
    if not I.ndim in (2, 3):
        raise Exception("'I' must be a 2-D or 3D array.")

    if BW.shape != I.shape:
        raise Exception("'I' and 'BW' must have the same shape.")

    if BW.dtype is not bool:
        BW = BW != 0

    # set default connectivity depending on whether the image is 2-D or 3-D
    if conn == None:
        if I.ndim == 3:
            conn = 26
        else:
            conn = 8
    else:
        if conn in (4, 8) and I.ndim == 3:
            raise Exception("'conn' is invalid for a 3-D image.")
        elif conn in (6, 18, 26) and I.ndim == 2:
            raise Exception("'conn' is invalid for a 2-D image.")

    # create structuring element depending on connectivity
    if conn == 4:
        selem = disk(1)
    elif conn == 8:
        selem = square(3)
    elif conn == 6:
        selem = ball(1)
    elif conn == 18:
        selem = ball(1)
        selem[:, 1, :] = 1
        selem[:, :, 1] = 1
        selem[1] = 1
    elif conn == 26:
        selem = cube(3)

    fm = I.astype(float)

    try:
        fm[BW]                 = -math.inf
        fm[np.logical_not(BW)] = math.inf
    except:
        fm[BW]                 = -float("inf")
        fm[np.logical_not(BW)] = float("inf")

    if I.dtype == float:
        I_range = np.amax(I) - np.amin(I)

        if I_range == 0:
            h = 0.1
        else:
            h = I_range*0.001
    else:
        h = 1

    fp1 = I + h

    g = np.minimum(fp1, fm)

    # perform reconstruction and get the image complement of the result
    if I.dtype == float:
        J = reconstruction(1 - fm, 1 - g, selem=selem)
        J = 1 - J
    else:
        J = reconstruction(255 - fm, 255 - g, method='dilation', selem=selem)
        J = 255 - J

    try:
        J[BW] = -math.inf
    except:
        J[BW] = -float("inf")

    return J


def closing_img(img):
    '''
    Image closing
    '''
    print("Image closing...")
    img[img!=0] = 1
    img_closed = multiBinaryClosing(img, 3)
    img_closed[img_closed!=0] = 255
    return img_closed


def water_shed(bw):
    '''
    Watershed segmentation
    '''
    print("Watershed segmentation...")
    bw[bw!=0] = 1
    D = -distance_transform_edt(bw)
    img = h_minima(D,2)
    lm = minimum_filter(img,size=2)
    mask = (img==lm)
    D2 = imimposemin(D,mask)
    Ld = watershedsNew(D2.astype(np.float32))
    seg_img = bw
    seg_img[Ld==0] = 0
    seg_img[seg_img!=0] = 255
    return seg_img


def main():
    img = tif_read('/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/pipeline_test/closed.tif')
    # img = closing_img(img)
    img = water_shed(img)
    tif_write(img, '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/pipeline_test/watershed.tif')

if __name__ == "__main__":
    main()
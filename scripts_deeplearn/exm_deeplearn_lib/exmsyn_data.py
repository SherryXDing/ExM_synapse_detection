from skimage import io
from skimage.measure import label, regionprops
from skimage.morphology import ball, closing
import numpy as np
import math
import random
import os
from scipy.ndimage.interpolation import rotate
import nrrd 


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


def _expand_img(img):
    """
    expand 3D image by concatenating the original in three dimensions
    """
    img = np.concatenate((img,img,img), axis=0)
    img = np.concatenate((img,img,img), axis=1)
    img = np.concatenate((img,img,img), axis=2)
    return img


def _unique_mask(mask):
    """
    check the uniqueness (only center object is masked) of a mask
    """
    label_mask = label(mask, neighbors=8)
    region_prop = regionprops(label_mask)
    if len(region_prop) > 1:
        for i in range(len(region_prop)):
            min_row, min_col, min_slice, max_row, max_col, max_slice = region_prop[i].bbox
            center = [min_row+math.floor((max_row-min_row)/2), min_col+math.floor((max_col-min_col)/2), min_slice+math.floor((max_slice-min_slice)/2)]
            dist = np.sqrt((center[0]-mask.shape[0]/2)**2 + (center[1]-mask.shape[1]/2)**2 + (center[2]-mask.shape[2]/2)**2)
            if dist > 5.0:
                mask[label_mask==i+1] = 0
    return mask


def _flip_img(img, xaxis_flip=False, zaxis_flip=False):
    """
    flip image in x-axis or z-axis.
    no y-axis flip because this can be done by rotating x-axis flipped images
    """
    if xaxis_flip or zaxis_flip:
        if xaxis_flip:
            img = np.flip(img, axis=0)
        if zaxis_flip:
            img = np.flip(img, axis=2)
    return img


# def _rot90_img(img, k=0):
#     """
#     rotate 3D image in x-y plane with k (k=1,2 or 3) times of 90 degrees
#     """
#     if k:
#         img = np.rot90(img, k)
#     return img


def _rotate_img(img, angle=0, is_mask=False):
    """
    rotate 3D image in x-y plane with a defined angle [0, 360)
    """
    if angle:
        rows, cols, slices = img.shape
        img = _expand_img(img)
        out_img = rotate(img, angle, axes=(1,0), reshape=False)
        out_img = out_img[rows:rows*2, cols:cols*2, slices:slices*2]
        if is_mask:
            out_img = _unique_mask(out_img)
    return out_img


def augment_small_sample(img, mask, agg_rotate=False):
    """
    do all types of augmentation for one image, use if the sample size is relatetively small
    x-flip, z-flip, x- and z-flip
    rotate 90, 180, 270 degrees (k=1,2,3)
    if doing aggressive rotation, rotate k (k=(1~23)) times of 15 degrees
    return a list of augmented images, including the original image 
    """
    aug_img=[]
    aug_mask=[]
    aug_img.append(img)
    aug_mask.append(mask)

    # x-flip
    xflip_img = _flip_img(img, xaxis_flip=True, zaxis_flip=False)
    xflip_mask = _flip_img(mask, xaxis_flip=True, zaxis_flip=False)
    aug_img.append(xflip_img)
    aug_mask.append(xflip_mask)
    # z-flip
    zflip_img = _flip_img(img, xaxis_flip=False, zaxis_flip=True)
    zflip_mask = _flip_img(mask, xaxis_flip=False, zaxis_flip=True)
    aug_img.append(zflip_img)
    aug_mask.append(zflip_mask)
    # x- and z-flip
    xzfilp_img = _flip_img(img, xaxis_flip=True, zaxis_flip=True)
    xzflip_mask = _flip_img(mask, xaxis_flip=True, zaxis_flip=True)
    aug_img.append(xzfilp_img)
    aug_mask.append(xzflip_mask)

    # rotate original, x-flipped, z-flipped, and xz-flipped images and masks
    if agg_rotate:
        k_all = tuple(range(1,24))
    else:
        k_all = (6,12,18)

    for k in k_all:
        # rotate original image and mask
        rot_orig_img = _rotate_img(img, angle=k*15, is_mask=False)
        rot_orig_mask = _rotate_img(mask, angle=k*15, is_mask=True)
        aug_img.append(rot_orig_img)
        aug_mask.append(rot_orig_mask)
        # rotate x-flipped image and mask
        rot_xflip_img = _rotate_img(xflip_img, angle=k*15, is_mask=False)
        rot_xflip_mask = _rotate_img(xflip_mask, angle=k*15, is_mask=True)
        aug_img.append(rot_xflip_img)
        aug_mask.append(rot_xflip_mask)
        # rotate z-flipped image and mask
        rot_zflip_img = _rotate_img(zflip_img, angle=k*15, is_mask=False)
        rot_zflip_mask = _rotate_img(zflip_mask, angle=k*15, is_mask=True)
        aug_img.append(rot_zflip_img)
        aug_mask.append(rot_zflip_mask)
        # rotate x- and z-flipped image and mask
        rot_xzflip_img = _rotate_img(xzfilp_img, angle=k*15, is_mask=False)
        rot_xzflip_mask = _rotate_img(xzflip_mask, angle=k*15, is_mask=True)
        aug_img.append(rot_xzflip_img)
        aug_mask.append(rot_xzflip_mask)

    return aug_img, aug_mask


def augment_random(imgs, masks, agg_rotate=False):
    """
    do random augmentation on two lists of images and corresponding masks, includes
    x-flip, z-flip, x- and z-flip
    rotate 90, 180, 270 degrees (k=1,2,3)
    if doing aggressive rotation (agg_rotate=True), rotate k (k=(1~23)) times of 15 degrees
    return two lists of augmented images and masks, with the same size as the input
    """
    assert len(imgs)==len(masks), \
        "Number of images and masks are not match!"
    sz = len(imgs)
    xflip = np.random.randint(2, size=sz)
    zflip = np.random.randint(2, size=sz)
    if agg_rotate:
        angle_all = np.random.randint(24, size=sz)*15
    else:
        angle_all = np.random.randint(4, size=sz)*90
    
    # random augmentation
    for i in range(sz):
        # x-flip or/and z-flip
        if xflip[i] or zflip[i]:
            imgs[i] = _flip_img(imgs[i], xaxis_flip=xflip[i], zaxis_flip=zflip[i])
            masks[i] = _flip_img(masks[i], xaxis_flip=xflip[i], zaxis_flip=zflip[i])
        # rotate
        if angle_all[i]:
            imgs[i] = _rotate_img(imgs[i], angle=angle_all[i])
            masks[i] = _rotate_img(masks[i], angle=angle_all[i], is_mask=True)
    
    return imgs, masks


def gen_pos_samples(img, mask, sz, out_path, expand=False):
    """
    generate positive samples from one mask that includes all objects
    img, mask: an image crop and corresponding mask of synapses
    sz: output sample size in (x,y,z)
    out_path: output folder directory
    expand: if expand the data in x-,y-,z- dimensions
    """
    assert img.shape == mask.shape, \
        "Image size and mask size are not match!"
    
    mask[mask!=255] = 0
    label_mask = label(mask, neighbors=8)
    region_prop = regionprops(label_mask)
    print("Total number of positive samples in current image: {}".format(len(region_prop)))

    for i in range(len(region_prop)):

        print("...Now processing positive sample {}".format(i))
        min_row, min_col, min_slice, max_row, max_col, max_slice = region_prop[i].bbox
        center = [min_row+math.floor((max_row-min_row)/2), min_col+math.floor((max_col-min_col)/2), min_slice+math.floor((max_slice-min_slice)/2)]
        
        if expand:
            new_img = _expand_img(img)
            new_mask = _expand_img(mask)
            new_center = [x+y for x,y in zip(center, [img.shape[0],img.shape[1],img.shape[2]])]
        elif (center[0]+sz[0]/2<img.shape[0] and center[0]-sz[0]/2>0
            and center[1]+sz[1]/2<img.shape[1] and center[1]-sz[1]/2>0
            and center[2]+sz[2]/2<img.shape[2] and center[2]-sz[2]/2>0):
            new_img = img
            new_mask = mask
            new_center = center
        else:
            print("......Failed! Current sample is too close to the edge.")
            continue
        
        sample_img = new_img[int(new_center[0]-sz[0]/2):int(new_center[0]+sz[0]/2), \
                    int(new_center[1]-sz[1]/2):int(new_center[1]+sz[1]/2), \
                    int(new_center[2]-sz[2]/2):int(new_center[2]+sz[2]/2)]
        sample_mask = new_mask[int(new_center[0]-sz[0]/2):int(new_center[0]+sz[0]/2), \
                    int(new_center[1]-sz[1]/2):int(new_center[1]+sz[1]/2), \
                    int(new_center[2]-sz[2]/2):int(new_center[2]+sz[2]/2)]
        sample_mask = _unique_mask(sample_mask)

        file_img =  out_path + "img_{}.tif".format(i)
        tif_write(sample_img, file_img)
        file_mask = out_path + "mask_{}.tif".format(i)
        tif_write(sample_mask, file_mask)

    return None


def gen_pos_samples_from_mask(img_name, mask_folder, sz, out_path, max_sz=56, expand=True):
    """
    generate positive samples using separated masks in a folder, each mask is a synapse
    img_name: name of an image crop
    mask_folder: directory of folder that includes masks of synapses, one mask for one synapse
    sz: output sample size in (x,y,z)
    out_path: output folder directory
    max_sz: biggest size of synapse that will be included; if bigger than this, then exclude the current junk sample
    expand: if expand the data in x-,y-,z- dimensions
    """
    assert os.path.exists(img_name), \
        "Image does not exist!"
    assert os.path.exists(mask_folder), \
        "Mask folder does not exist!"
    
    img, head = nrrd.read(img_name)
    all_masks = os.listdir(mask_folder)
    sample_idx = 0

    for mask_file in all_masks:
        mask, head = nrrd.read(mask_folder+mask_file)
        assert img.shape == mask.shape, \
            "Image size and "+mask_file+" size are not match!"

        mask[mask!=0] = 255
        label_mask = label(mask, neighbors=8)
        region_prop = regionprops(label_mask)
        if len(region_prop) != 1:
            print(mask_file+" includes more than one synapse, please double check!")
            continue

        min_row, min_col, min_slice, max_row, max_col, max_slice = region_prop[0].bbox
        center = [min_row+math.floor((max_row-min_row)/2), min_col+math.floor((max_col-min_col)/2), min_slice+math.floor((max_slice-min_slice)/2)]
        
        if (max_row-min_row>max_sz or max_col-min_col>max_sz or max_slice-min_slice>max_sz):
            continue

        if expand:
            new_img = _expand_img(img)
            new_mask = _expand_img(mask)
            new_center = [x+y for x,y in zip(center, [img.shape[0],img.shape[1],img.shape[2]])]
        elif (center[0]+sz[0]/2<img.shape[0] and center[0]-sz[0]/2>0
            and center[1]+sz[1]/2<img.shape[1] and center[1]-sz[1]/2>0
            and center[2]+sz[2]/2<img.shape[2] and center[2]-sz[2]/2>0):
            new_img = img
            new_mask = mask
            new_center = center
        else:
            print("......Failed! Current sample is too close to the edge.")
            continue
        
        sample_img = new_img[int(new_center[0]-sz[0]/2):int(new_center[0]+sz[0]/2), \
                    int(new_center[1]-sz[1]/2):int(new_center[1]+sz[1]/2), \
                    int(new_center[2]-sz[2]/2):int(new_center[2]+sz[2]/2)]
        sample_mask = new_mask[int(new_center[0]-sz[0]/2):int(new_center[0]+sz[0]/2), \
                    int(new_center[1]-sz[1]/2):int(new_center[1]+sz[1]/2), \
                    int(new_center[2]-sz[2]/2):int(new_center[2]+sz[2]/2)]

        file_img = out_path + "img_{}.nrrd".format(sample_idx)
        nrrd.write(file_img, sample_img)
        file_mask = out_path + "mask_{}.nrrd".format(sample_idx)
        nrrd.write(file_mask, sample_mask)
        sample_idx += 1
    
    return None


def gen_pos_samples_overlapped_maxima(img_name, mask_folder, local_maxima_name, sz, out_path, expand=True):
    """
    generate positive samples that have less than 5 local maxima, using separated masks in a folder, each mask is a synapse
    img_name: name of an image crop
    mask_folder: directory of folder that includes masks of synapses, one mask for one synapse
    local_maxima_name: name of local maxima image
    sz: output sample size in (x,y,z)
    out_path: output folder directory
    expand: if expand the data in x-,y-,z- dimensions
    """
    assert os.path.exists(img_name), \
        "Image does not exist!"
    assert os.path.exists(mask_folder), \
        "Mask folder does not exist!"
    assert os.path.exists(local_maxima_name), \
        "Local maxima image does not exist!"
    
    img, head = nrrd.read(img_name)
    local_maxima_img, head = nrrd.read(local_maxima_name)
    assert img.shape == local_maxima_img.shape, \
        "Image size and local maxima image size are not match!"
    all_masks = os.listdir(mask_folder)

    for mask_file in all_masks:
        mask, head = nrrd.read(mask_folder+mask_file)
        assert img.shape == mask.shape, \
            "Image size and "+mask_file+" size are not match!"

        mask[mask!=0] = 1
        curr_maxima = mask * local_maxima_img
        if np.count_nonzero(curr_maxima) > 4:
            continue

        label_mask = label(mask, neighbors=8)
        region_prop = regionprops(label_mask)
        if len(region_prop) != 1:
            print(mask_file+" includes more than one synapse, please double check!")
            continue

        min_row, min_col, min_slice, max_row, max_col, max_slice = region_prop[0].bbox
        center = [min_row+math.floor((max_row-min_row)/2), min_col+math.floor((max_col-min_col)/2), min_slice+math.floor((max_slice-min_slice)/2)]
        
        if (max_row-min_row>sz[0] or max_col-min_col>sz[1] or max_slice-min_slice>sz[2]):
            continue

        if expand:
            new_img = _expand_img(img)
            new_mask = _expand_img(mask)
            new_center = [x+y for x,y in zip(center, [img.shape[0],img.shape[1],img.shape[2]])]
        elif (center[0]+sz[0]/2<img.shape[0] and center[0]-sz[0]/2>0
            and center[1]+sz[1]/2<img.shape[1] and center[1]-sz[1]/2>0
            and center[2]+sz[2]/2<img.shape[2] and center[2]-sz[2]/2>0):
            new_img = img
            new_mask = mask
            new_center = center
        else:
            print("......Failed! Current sample is too close to the edge.")
            continue
        
        sample_img = new_img[int(new_center[0]-sz[0]/2):int(new_center[0]+sz[0]/2), \
                    int(new_center[1]-sz[1]/2):int(new_center[1]+sz[1]/2), \
                    int(new_center[2]-sz[2]/2):int(new_center[2]+sz[2]/2)]
        sample_mask = new_mask[int(new_center[0]-sz[0]/2):int(new_center[0]+sz[0]/2), \
                    int(new_center[1]-sz[1]/2):int(new_center[1]+sz[1]/2), \
                    int(new_center[2]-sz[2]/2):int(new_center[2]+sz[2]/2)]
        sample_img = sample_img * sample_mask

        file_img = out_path + mask_file
        nrrd.write(file_img, sample_img)
    
    return None


def gen_neg_samples(img, mask, sz, out_path):
    """
    generate negative samples, if not enough samples, generate negative samples by randomly cropping the image
    img, mask: an image crop and corresponding mask of synapses
    sz: output sample size in (x,y,z)
    out_path: output folder directory
    """
    assert img.shape == mask.shape, \
        "Image size and mask size are not match!"
    
    syn_mask = np.zeros(mask.shape, dtype=mask.dtype)
    syn_mask[:,:,:] = mask
    syn_mask[syn_mask!=255]=0
    label_syn = label(syn_mask, neighbors=8)
    regprop_syn = regionprops(label_syn)
    total_num = len(regprop_syn)
    print("For balanced sample sizes, total number of negative samples to be created is {}".format(total_num))
  
    junk_mask = np.zeros(mask.shape, dtype=mask.dtype)
    junk_mask[:,:,:] = mask
    junk_mask[junk_mask==255]=0
    junk_mask = closing(junk_mask, ball(6))  # image closing on junk pieces, not necessary for preprocessed masks
    label_junk = label(junk_mask, neighbors=8)
    regprop_junk = regionprops(label_junk)
    
    curr_num = 0
    # each sample includes a junk in the center
    for i in range(len(regprop_junk)):
        if curr_num < total_num: 
            min_row, min_col, min_slice, max_row, max_col, max_slice = regprop_junk[i].bbox
            center = [min_row+math.floor((max_row-min_row)/2), min_col+math.floor((max_col-min_col)/2), min_slice+math.floor((max_slice-min_slice)/2)]
            if (center[0]+sz[0]/2<img.shape[0] and center[0]-sz[0]/2>0
                and center[1]+sz[1]/2<img.shape[1] and center[1]-sz[1]/2>0
                and center[2]+sz[2]/2<img.shape[2] and center[2]-sz[2]/2>0):
                sample_img = img[int(center[0]-sz[0]/2):int(center[0]+sz[0]/2), \
                            int(center[1]-sz[1]/2):int(center[1]+sz[1]/2), int(center[2]-sz[2]/2):int(center[2]+sz[2]/2)]
                sample_mask = mask[int(center[0]-sz[0]/2):int(center[0]+sz[0]/2), \
                            int(center[1]-sz[1]/2):int(center[1]+sz[1]/2), int(center[2]-sz[2]/2):int(center[2]+sz[2]/2)]
                sample_mask[sample_mask!=255]=0
                file_img = out_path + "img_{}.tif".format(curr_num)
                tif_write(sample_img, file_img)
                file_mask = out_path + "mask_{}.tif".format(curr_num)
                tif_write(sample_mask, file_mask)
                print("...Now prepared {} negative samples".format(curr_num))
                curr_num += 1
    
    # if there are not enough junks as negative samples, randomly crop the image
    while curr_num < total_num:
        # pick a random voxel
        min_row = random.randint(0, img.shape[0]-sz[0]-1)
        min_col = random.randint(0, img.shape[1]-sz[1]-1)
        min_slice = random.randint(0, img.shape[2]-sz[2]-1)
        sample_img = img[min_row:min_row+sz[0], min_col:min_col+sz[1], min_slice:min_slice+sz[2]]
        sample_mask = mask[min_row:min_row+sz[0], min_col:min_col+sz[1], min_slice:min_slice+sz[2]]
        is_synapse = False
        sample_mask[sample_mask!=255]=0
        if np.amax(sample_mask)==255:  # if there are synapses in the cropped sample, make sure it's not in the center area
            label_sample = label(sample_mask, neighbors=8)
            regprop_sample = regionprops(label_sample)
            for j in range(len(regprop_sample)):
                min_r, min_c, min_s, max_r, max_c, max_s = regprop_sample[j].bbox
                syn_center = [min_r+math.floor((max_r-min_r)/2), min_c+math.floor((max_c-min_c)/2), min_s+math.floor((max_s-min_s)/2)]
                dist = np.sqrt((syn_center[0]-sz[0]/2)**2 + (syn_center[1]-sz[1]/2)**2 + (syn_center[2]-sz[2]/2)**2)
                if dist < 18.0:
                    is_synapse = True
        if is_synapse:
            continue
        file_img = out_path + "img_{}.tif".format(curr_num)
        tif_write(sample_img, file_img)
        file_mask = out_path + "mask_{}.tif".format(curr_num)
        tif_write(sample_mask, file_mask)
        print("...Now prepared {} negative samples".format(curr_num))
        curr_num += 1

    return None


def gen_neg_samples_from_mask(img_name, mask_name, sz, out_path, max_sz=56, expand=True):
    """
    generate negative samples using a junk mask
    img_name, mask_name: names of an image crop and corresponding mask of junks
    sz: output sample size in (x,y,z)
    out_path: output folder directory
    max_sz: biggest size of junks that will be included; if bigger than this, then exclude the current junk sample
    expand: if expand the data in x-,y-,z- dimensions
    """
    assert os.path.exists(img_name), \
        "Image does not exist!"
    assert os.path.exists(mask_name), \
        "Mask does not exist!"

    img, head = nrrd.read(img_name)
    mask, head = nrrd.read(mask_name)
    assert img.shape == mask.shape, \
        "Image size and mask size are not match!"
    
    label_junk = label(mask, neighbors=8)
    regprop_junk = regionprops(label_junk)
    sample_idx = 0

    # each sample includes a junk in the center
    for i in range(len(regprop_junk)):
        min_row, min_col, min_slice, max_row, max_col, max_slice = regprop_junk[i].bbox
        center = [min_row+math.floor((max_row-min_row)/2), min_col+math.floor((max_col-min_col)/2), min_slice+math.floor((max_slice-min_slice)/2)]

        if (max_row-min_row>max_sz or max_col-min_col>max_sz or max_slice-min_slice>max_sz):
            continue

        if expand:
            new_img = _expand_img(img)
            new_mask = _expand_img(mask)
            new_center = [x+y for x,y in zip(center, [img.shape[0],img.shape[1],img.shape[2]])]
        elif (center[0]+sz[0]/2<img.shape[0] and center[0]-sz[0]/2>0
            and center[1]+sz[1]/2<img.shape[1] and center[1]-sz[1]/2>0
            and center[2]+sz[2]/2<img.shape[2] and center[2]-sz[2]/2>0):
            new_img = img
            new_mask = mask
            new_center = center
        else:
            print("......Failed! Current sample is too close to the edge.")
            continue
            
        sample_img = new_img[int(new_center[0]-sz[0]/2):int(new_center[0]+sz[0]/2), \
                int(new_center[1]-sz[1]/2):int(new_center[1]+sz[1]/2), \
                int(new_center[2]-sz[2]/2):int(new_center[2]+sz[2]/2)]
        sample_mask = new_mask[int(new_center[0]-sz[0]/2):int(new_center[0]+sz[0]/2), \
                int(new_center[1]-sz[1]/2):int(new_center[1]+sz[1]/2), \
                int(new_center[2]-sz[2]/2):int(new_center[2]+sz[2]/2)]
        sample_mask = _unique_mask(sample_mask)

        file_img = out_path + "img_{}.nrrd".format(sample_idx)
        nrrd.write(file_img, sample_img)
        file_mask = out_path + "mask_{}.nrrd".format(sample_idx)
        nrrd.write(file_mask, sample_mask)
        sample_idx += 1

    return None


def gen_neg_samples_overlapped_maxima(img_name, mask_name, local_maxima_name, sz, out_path, expand=True):
    """
    generate negative samples that have more than 1 local maxima, using a junk mask
    img_name, mask_name: names of an image crop and corresponding mask of junks
    local_maxima_name: name of local maxima image
    sz: output sample size in (x,y,z)
    out_path: output folder directory
    expand: if expand the data in x-,y-,z- dimensions
    """
    assert os.path.exists(img_name), \
        "Image does not exist!"
    assert os.path.exists(mask_name), \
        "Mask does not exist!"
    assert os.path.exists(local_maxima_name), \
        "Local maxima image does not exist!"

    img, head = nrrd.read(img_name)
    mask, head = nrrd.read(mask_name)
    local_maxima_img, head = nrrd.read(local_maxima_name)

    assert img.shape == mask.shape, \
        "Image size and mask size are not match!"
    assert img.shape == local_maxima_img.shape, \
        "Image size and local maxima image size are not match!"
    
    label_junk = label(mask, neighbors=8)
    regprop_junk = regionprops(label_junk)
    sample_idx = 0

    # each sample includes a junk in the center
    for i in range(len(regprop_junk)):
        curr_junk = np.zeros(mask.shape, dtype=mask.dtype)
        curr_junk[label_junk==regprop_junk[i].label] = 1
        curr_maxima = curr_junk * local_maxima_img
        if np.count_nonzero(curr_maxima) < 2:
            continue

        min_row, min_col, min_slice, max_row, max_col, max_slice = regprop_junk[i].bbox
        center = [min_row+math.floor((max_row-min_row)/2), min_col+math.floor((max_col-min_col)/2), min_slice+math.floor((max_slice-min_slice)/2)]

        if (max_row-min_row>sz[0] or max_col-min_col>sz[1] or max_slice-min_slice>sz[2]):
            continue

        if expand:
            new_img = _expand_img(img)
            new_mask = _expand_img(curr_junk)
            new_center = [x+y for x,y in zip(center, [img.shape[0],img.shape[1],img.shape[2]])]
        elif (center[0]+sz[0]/2<img.shape[0] and center[0]-sz[0]/2>0
            and center[1]+sz[1]/2<img.shape[1] and center[1]-sz[1]/2>0
            and center[2]+sz[2]/2<img.shape[2] and center[2]-sz[2]/2>0):
            new_img = img
            new_mask = curr_junk
            new_center = center
        else:
            print("......Failed! Current sample is too close to the edge.")
            continue
            
        sample_img = new_img[int(new_center[0]-sz[0]/2):int(new_center[0]+sz[0]/2), \
                int(new_center[1]-sz[1]/2):int(new_center[1]+sz[1]/2), \
                int(new_center[2]-sz[2]/2):int(new_center[2]+sz[2]/2)]
        sample_mask = new_mask[int(new_center[0]-sz[0]/2):int(new_center[0]+sz[0]/2), \
                int(new_center[1]-sz[1]/2):int(new_center[1]+sz[1]/2), \
                int(new_center[2]-sz[2]/2):int(new_center[2]+sz[2]/2)]
        sample_img = sample_img * sample_mask

        file_img = out_path + "junk_{}.nrrd".format(sample_idx)
        nrrd.write(file_img, sample_img)
        sample_idx += 1

    return None


def main():
    """
    generate positive and negative samples
    """
    sz = (48,48,48)
    
    # img_name = input("Enter image name: ")
    # while not os.path.exists(img_name):
    #     print("...No such image, try again...")
    #     img_name = input("Enter image name: ")        

    # pos_mask_folder = input("Enter mask folder for positive samples (end with '/'): ")
    # while not os.path.exists(pos_mask_folder):
    #     print("...No such mask folder, try again...")
    #     pos_mask_folder = input("Enter mask folder for positive samples (end with '/'): ")  

    # path_pos = input("Enter output directory for positive samples (end with '/'): ")
    # while not os.path.exists(path_pos):
    #     print("...No such folder, creating now...")
    #     os.mkdir(path_pos, mode=0o755)    

    # neg_mask_name = input("Enter negative sample mask name: ")
    # while not os.path.exists(neg_mask_name):
    #     print("...No such mask image, try again...")
    #     neg_mask_name = input("Enter negative sample mask name: ")

    # path_neg = input("Enter output directory for negative samples (end with '/'): ")
    # while not os.path.exists(path_neg):
    #     print("...No such folder, creating now...")
    #     os.mkdir(path_neg) 
    
    # gen_pos_samples_from_mask(img_name, pos_mask_folder, sz, path_pos)
    # gen_neg_samples_from_mask(img_name, neg_mask_name, sz, path_neg)

    file_path = "/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/"
    sub_folder = ["antennal_lobe_1/", "ellipsoid_body_1/", "mushroom_body_1/", "optic_lobe_1/", "protocerebrum_1/"]
    img_all = ["5793_2782_11411-background_subtract.nrrd", "C1-7527_3917_6681-background_subtract.nrrd", \
        "C1-6210_6934_5492-background_subtract.nrrd", "C1-4228_3823_4701-background_subtract.nrrd", "C1-13904_10064_4442-background_subtract.nrrd"]
    
    for i in range(len(sub_folder)):
        img_name = file_path + sub_folder[i] + img_all[i]
        pos_mask_folder = file_path + sub_folder[i] + "synapses_final/" 
        local_maxima_name = file_path + sub_folder[i] + "local_maxima_3d.nrrd"
        pos_out_path = file_path + sub_folder[i] + "samples_overlapped_maxima/pos/"
        gen_pos_samples_overlapped_maxima(img_name=img_name, mask_folder=pos_mask_folder, local_maxima_name=local_maxima_name, sz=sz, out_path=pos_out_path)

        junk_mask_name = file_path + sub_folder[i] + "mask_junk.nrrd"
        neg_out_path = file_path + sub_folder[i] + "samples_overlapped_maxima/neg/"
        gen_neg_samples_overlapped_maxima(img_name=img_name, mask_name=junk_mask_name, local_maxima_name=local_maxima_name, sz=sz, out_path=neg_out_path)

    print("Done!")

if __name__ == "__main__":
    main()
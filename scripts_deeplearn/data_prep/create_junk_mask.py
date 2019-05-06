import nrrd
import os 
import numpy as np 
from skimage.filters import threshold_triangle
from skimage.measure import label, regionprops 
from skimage.morphology import ball, closing


def combine_masks(file_path):
    """
    combine all synapse masks into one mask
    """
    assert os.path.exists(file_path), \
        "Folder does not exist!"

    all_files = os.listdir(file_path)
    mask, head = nrrd.read(file_path+all_files[0])
    mask_all = np.zeros(mask.shape, dtype=mask.dtype)
    for curr_file in all_files:
        mask, head = nrrd.read(file_path+curr_file)
        mask_all[mask!=0] = 1

    return mask_all


def gen_junk_mask(mask_edge_synapse, img_thresholded):
    """
    generate a mask including all junks
    """
    labeled_img = label(img_thresholded, neighbors=8)
    region_prop = regionprops(labeled_img)
    mask_junk = np.zeros(mask_edge_synapse.shape, dtype=mask_edge_synapse.dtype)
    for i in range(len(region_prop)):
        if region_prop[i].area > 250:
            curr_label = region_prop[i].label
            if np.amax(mask_edge_synapse[labeled_img==curr_label]) != 255:
                mask_junk[labeled_img==curr_label] = 255
    
    mask_junk = closing(mask_junk, ball(4))  # image closing on junk pieces
    mask_junk[mask_junk!=0] = 255

    return mask_junk


def main():

    folder_path = "/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/antennal_lobe_1/"

    """generate a mask including all labeled synapses and edges"""
    img_edge_subtract, head = nrrd.read(folder_path+"5793_2782_11411-edgesubtract.nrrd")
    mask_edge_synapse = np.zeros(img_edge_subtract.shape, dtype=img_edge_subtract.dtype)
    mask_edge_synapse[img_edge_subtract==0] = 255
    nrrd.write(folder_path+'mask_edge.nrrd', mask_edge_synapse)
    mask_synapse = combine_masks(folder_path+"synapses_final/")
    mask_synapse[mask_synapse!=0] = 255
    nrrd.write(folder_path+'mask_synapse.nrrd', mask_synapse)
    mask_edge_synapse[mask_synapse!=0] = 255
    nrrd.write(folder_path+"mask_edge_synapse.nrrd", mask_edge_synapse)
    # mask_edge_synapse, head = nrrd.read(folder_path+"mask_edge_synapse.nrrd")

    """segment background subtracted image using threshold value based on the triangle algorithm"""
    # img_bg_subtract, head = nrrd.read(folder_path+"C1-6210_6934_5492-background_subtract.nrrd")
    # thres = threshold_triangle(img_bg_subtract)
    # img_thresholded = np.zeros(img_bg_subtract.shape, dtype=img_bg_subtract.dtype)
    # img_thresholded[img_bg_subtract>thres] = 255
    # nrrd.write(folder_path+"img_thresholded.nrrd", img_thresholded)
    img_thresholded, head = nrrd.read(folder_path+"img_thresholded.nrrd")

    mask_junk = gen_junk_mask(mask_edge_synapse, img_thresholded)
    nrrd.write(folder_path+"mask_junk.nrrd", mask_junk)

    print('Done!')

    # folder_path = "/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/For_check/Segmented_mask/"
    # mask_synapse = combine_masks(folder_path+'all_synapses_with_corrections/')
    # mask_synapse[mask_synapse!=0] = 255
    # nrrd.write(folder_path+'mask_synapse_all.nrrd', mask_synapse)
    # print('Done!')


if __name__=="__main__":
    main()
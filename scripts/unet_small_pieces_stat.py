from scipy.io import loadmat
import numpy as np 
import os 
import nrrd


file_path = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/For_check/'
file_names = [file_path+'unet_results/'+f for f in os.listdir(file_path+'unet_results/') if f.endswith('.nrrd')]
data = loadmat(file_path+'example_very_small_U-net_masks_to_remove.mat')
mask_list = data['mask_list']

all_vxl = []
mask_vxl = []

for name in file_names:
    mask, head = nrrd.read(name)
    nonzero_vxl = np.count_nonzero(mask)
    base_name = os.path.basename(name)
    file_idx = os.path.splitext(base_name)[0]
    if int(file_idx) in mask_list:
        mask_vxl += [nonzero_vxl]
        print("Piece {}: {} voxels".format(file_idx, nonzero_vxl))
    else:
        all_vxl += [nonzero_vxl]

print("All masks:")
print("Min number of voxels {}".format(min(all_vxl)))
print("Max number of voxels {}".format(max(all_vxl)))

print("Small pieces:")
print("Min number of voxels {}".format(min(mask_vxl)))
print("Max number of voxels {}".format(max(mask_vxl)))
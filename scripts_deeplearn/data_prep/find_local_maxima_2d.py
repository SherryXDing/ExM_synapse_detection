from skimage.feature import peak_local_max
import nrrd 
import numpy as np 


# Part 1
file_path = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/mushroom_body_1/'
img, head = nrrd.read(file_path+'C1-6210_6934_5492-background_subtract.nrrd')

local_max_2d = np.zeros(img.shape, dtype=img.dtype)
for i in range(img.shape[2]):
    curr_img = img[:,:,i]
    curr_local_max = np.zeros(curr_img.shape, dtype=curr_img.dtype)
    idx = peak_local_max(curr_img, min_distance=2, threshold_abs=150, exclude_border=True, indices=False)
    curr_local_max[idx==True] = 255
    local_max_2d[:,:,i] = curr_local_max
nrrd.write(file_path+"local_maxima_2d.nrrd", local_max_2d)


# Part 2
file_path_all = ['/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/antennal_lobe_1/', \
    '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/ellipsoid_body_1/', \
    '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/mushroom_body_1/', \
    '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/optic_lobe_1/', \
    '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/protocerebrum_1/']

for i in range(len(file_path_all)):
    local_maxima_img, head  = nrrd.read(file_path_all[i]+"local_maxima_2d.nrrd")
    mask_junk, head = nrrd.read(file_path_all[i]+"mask_junk.nrrd")
    mask_junk[mask_junk!=0] = 1
    local_maxima_junk = local_maxima_img * mask_junk
    nrrd.write(file_path_all[i]+"local_maxima_junk.nrrd", local_maxima_junk)
    mask_synapse, head = nrrd.read(file_path_all[i]+"mask_synapse.nrrd")
    mask_synapse[mask_synapse!=0] = 1
    local_maxima_synapse = local_maxima_img * mask_synapse
    nrrd.write(file_path_all[i]+"local_maxima_synapse.nrrd", local_maxima_synapse)

print("Done!")
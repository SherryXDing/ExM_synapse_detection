import nrrd 
import os
import numpy as np 
from skimage.measure import label
import matplotlib.pyplot as plt 
from scipy.spatial import distance_matrix


data_path = ['/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/antennal_lobe_1/',
    '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/ellipsoid_body_1/',
    '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/mushroom_body_1/',
    '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/optic_lobe_1/',
    '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/protocerebrum_1/']

num_maxima_syn = []
num_maxima_junk = []
mean_intensity_syn = []
mean_intensity_junk = []
mean_dist_syn = []
mean_dist_junk = []

for i in range(len(data_path)):

    maxima_img, head = nrrd.read(data_path[i]+'local_maxima_3d.nrrd')
    all_syn = os.listdir(data_path[i]+'synapses_final/')
    n_max_syn = np.zeros((len(all_syn),1), dtype=int)
    intensity_syn = np.zeros((len(all_syn),1))
    dist_syn = np.zeros((len(all_syn),1))
    
    for j in range(len(all_syn)):
#        curr_syn, head = nrrd.read(data_path[i]+'synapses_final/'+all_syn[j])
#        curr_syn[curr_syn!=0] = 1
#        curr_maxima = curr_syn * maxima_img
#        nrrd.write(data_path[i]+'maxima/synapse/'+all_syn[j], curr_maxima)
        curr_maxima, head = nrrd.read(data_path[i]+'maxima/synapse/'+all_syn[j])
        # Num of local maxima
        n_max_syn[j] = np.count_nonzero(curr_maxima)
        # Mean intensity of local maxima
        if n_max_syn[j] != 0:
            intensity_syn[j] = curr_maxima.sum() / n_max_syn[j]
        else:
            intensity_syn[j] = 0
        # Mean distance
        if n_max_syn[j] > 1:
            xyz = np.transpose(np.nonzero(curr_maxima))
            dist = distance_matrix(xyz, xyz)
            dist_syn[j] = dist.sum() / (n_max_syn[j]*n_max_syn[j]-n_max_syn[j])
        else:
            dist_syn[j] = 0
        
    num_maxima_syn.extend(n_max_syn)
    mean_intensity_syn.extend(intensity_syn)
    mean_dist_syn.extend(dist_syn)

    all_junk, head = nrrd.read(data_path[i]+'mask_junk.nrrd')
    data_label = label(all_junk, neighbors=8)
    n_max_junk = np.zeros((np.amax(data_label),1), dtype=int)
    intensity_junk = np.zeros((np.amax(data_label),1))
    dist_junk = np.zeros((np.amax(data_label),1))
    
    for j in range(np.amax(data_label)):
#        curr_junk = np.zeros(all_junk.shape, dtype=all_junk.dtype)
#        curr_junk[data_label==j+1] = 1
#        curr_maxima = curr_junk * maxima_img
#        nrrd.write(data_path[i]+'maxima/junk/'+str(j)+'.nrrd', curr_maxima)
        curr_maxima, head = nrrd.read(data_path[i]+'maxima/junk/'+str(j)+'.nrrd')
        # Num of local maxima
        n_max_junk[j] = np.count_nonzero(curr_maxima)
        # Mean intensity of local maxima
        if n_max_junk[j] != 0:
            intensity_junk[j] = curr_maxima.sum() / n_max_junk[j]
        else:
            intensity_junk[j] = 0
        # Mean distance
        if n_max_junk[j] > 1:
            xyz = np.transpose(np.nonzero(curr_maxima))
            dist = distance_matrix(xyz, xyz)
            dist_junk[j] = dist.sum() / (n_max_junk[j]*n_max_junk[j]-n_max_junk[j])
        else:
            dist_junk[j] = 0
        
    num_maxima_junk.extend(n_max_junk)
    mean_intensity_junk.extend(intensity_junk)
    mean_dist_junk.extend(dist_junk)

num_maxima = [np.asarray(num_maxima_syn), np.asarray(num_maxima_junk)]
mean_intensity = [np.asarray(mean_intensity_syn), np.asarray(mean_intensity_junk)]
mean_dist = [np.asarray(mean_dist_syn), np.asarray(mean_dist_junk)]
fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()
label = ['synapse', 'junk']

ax0.hist(num_maxima, bins='auto', label=label)
ax0.legend(prop={'size': 10})
ax0.set_title('Num local maxima')
ax0.set_xlim(right=40)

ax1.hist(mean_intensity, bins='auto', label=label)
ax1.legend(prop={'size':10})
ax1.set_title('Mean intensity')

ax2.hist(mean_dist, bins='auto', label=label)
ax2.legend(prop={'size':10})
ax2.set_title('Mean distance')

ax3.scatter(np.asarray(num_maxima_syn), np.asarray(mean_dist_syn), color='r')
ax3.scatter(np.asarray(num_maxima_junk), np.asarray(mean_dist_junk), color='g')
ax3.set_xlabel('Num Maxima')
ax3.set_ylabel('Mean distance')
ax3.legend(label)

plt.tight_layout()
plt.savefig('/groups/scicompsoft/home/dingx/Documents/ExM/scripts/stats_local_maxima.pdf')
plt.show()
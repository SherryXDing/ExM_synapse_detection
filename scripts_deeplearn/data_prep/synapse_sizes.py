from data_check import check_syn_size, check_junk_size, xyz_size
#from data_check import min_max_size, list_large_synapse
import numpy as np 
import matplotlib.pyplot as plt

# syn_size_optic = check_syn_size("/groups/scicompsoft/home/dingx/Documents/ExM/data/optic_lobe_1/synapses_final/")
# syn_size_ellipsoid = check_syn_size("/groups/scicompsoft/home/dingx/Documents/ExM/data/ellipsoid_body_1/synapses_final/")
# syn_size_protocerebrum = check_syn_size("/groups/scicompsoft/home/dingx/Documents/ExM/data/protocerebrum_1/synapses_final/")

# syn_optic = xyz_size(syn_size_optic)
# syn_ellipsoid = xyz_size(syn_size_ellipsoid)
# syn_protocerebrum = xyz_size(syn_size_protocerebrum)
# syn_all = np.concatenate((syn_optic, syn_ellipsoid, syn_protocerebrum))

# min_optic, max_optic = min_max_size(syn_size_optic)
# min_ellipsoid, max_ellipsoid = min_max_size(syn_size_ellipsoid)
# min_protocerebrum, max_protocerebrum = min_max_size(syn_size_protocerebrum)

# large_optic = list_large_synapse(syn_size_optic)
# large_ellipsoid = list_large_synapse(syn_size_ellipsoid)
# large_protocerebrum = list_large_synapse(syn_size_protocerebrum)

synapse_path = ["/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/antennal_lobe_1/synapses_final/", \
    "/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/ellipsoid_body_1/synapses_final/", \
    "/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/mushroom_body_1/synapses_final/", \
    "/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/optic_lobe_1/synapses_final/", \
    "/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/protocerebrum_1/synapses_final/"]

syn_all = []
for i in range(len(synapse_path)):
    syn_size = check_syn_size(synapse_path[i])
    syn_i = xyz_size(syn_size)
    if i == 0:
        syn_all = syn_i
    else:
        syn_all = np.concatenate((syn_all, syn_i))

# junk_size_optic = check_junk_size("/groups/scicompsoft/home/dingx/Documents/ExM/data/optic_lobe_1/mask_junk.nrrd")
# junk_size_ellipsoid = check_junk_size("/groups/scicompsoft/home/dingx/Documents/ExM/data/ellipsoid_body_1/mask_junk.nrrd")
# junk_size_protocerebrum = check_junk_size("/groups/scicompsoft/home/dingx/Documents/ExM/data/protocerebrum_1/mask_junk.nrrd")

# junk_optic = xyz_size(junk_size_optic)
# junk_ellipsoid = xyz_size(junk_size_ellipsoid)
# junk_protocerebrum = xyz_size(junk_size_protocerebrum)
# junk_all = np.concatenate((junk_optic, junk_ellipsoid, junk_protocerebrum))

junk_file = ["/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/antennal_lobe_1/mask_junk.nrrd", \
    "/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/ellipsoid_body_1/mask_junk.nrrd", \
    "/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/mushroom_body_1/mask_junk.nrrd", \
    "/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/optic_lobe_1/mask_junk.nrrd", \
    "/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/protocerebrum_1/mask_junk.nrrd"]

junk_all = []
for j in range(len(junk_file)):
    junk_size = check_junk_size(junk_file[j])
    junk_j = xyz_size(junk_size)
    if j == 0:
        junk_all = junk_j
    else:
        junk_all = np.concatenate((junk_all, junk_j))

vx_size = [syn_all[:,0], junk_all[:,0]]
x_size = [syn_all[:,1], junk_all[:,1]]
y_size = [syn_all[:,2], junk_all[:,2]]
z_size = [syn_all[:,3], junk_all[:,3]]

fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()
label = ['syn','junk']
ax0.hist(vx_size, bins='auto', label=label)
ax0.legend(prop={'size': 10})
ax0.set_title('Num voxels')

ax1.hist(x_size, bins='auto', label=label)
ax1.legend(prop={'size': 10})
ax1.set_title('Size along x-axis')

ax2.hist(y_size, bins='auto', label=label)
ax2.legend(prop={'size': 10})
ax2.set_title('Size along y-axis')

ax3.hist(z_size, bins='auto', label=label)
ax3.legend(prop={'size': 10})
ax3.set_title('Size along z-axis')

plt.tight_layout()
plt.savefig('size_bars.pdf')
plt.show()
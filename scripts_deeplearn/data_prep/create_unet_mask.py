import nrrd

folder_path = ["/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/antennal_lobe_1/", \
               "/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/ellipsoid_body_1/", \
               "/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/mushroom_body_1/", \
               "/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/optic_lobe_1/", \
               "/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/protocerebrum_1/"]


for i in range(len(folder_path)):
    mask, head = nrrd.read(folder_path[i]+"mask_synapse.nrrd")
    mask[mask!=0]=1
    mask_edge, head = nrrd.read(folder_path[i]+"mask_edge.nrrd")
    mask[mask_edge!=0]=2
    nrrd.write(folder_path[i]+"mask_unet_syn1_edge2.nrrd", mask)

print("done!")
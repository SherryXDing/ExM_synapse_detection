import nrrd 
import os
import numpy as np 

file_path = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/test/results/junk/'

junk, head = nrrd.read(file_path+'junk_stats_0.nrrd')
junk_all = np.zeros(junk.shape, dtype=junk.dtype)
for file_i in os.listdir(file_path):
    junk, head = nrrd.read(file_path+file_i)
    junk_all += junk
nrrd.write(file_path+'junk_final_result.nrrd', junk_all)
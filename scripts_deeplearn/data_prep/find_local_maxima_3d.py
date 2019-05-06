import nrrd 
import numpy as np 


def find_local_max(img, min_distance=2, threshold_abs=150, normalization=False):
    if normalization:
        img = (img - img.mean()) / img.std()

    img_max = np.zeros(img.shape, dtype=img.dtype)
    for i in range(min_distance, img.shape[0]-min_distance):
        for j in range(min_distance, img.shape[1]-min_distance):
            for k in range(min_distance, img.shape[2]-min_distance):
                img_ijk = img[i-min_distance:i+min_distance+1, j-min_distance:j+min_distance+1, k-min_distance:k+min_distance+1]

                if img[i,j,k] == img_ijk.max() and img[i,j,k] > threshold_abs:
                    img_max[i,j,k] = img[i,j,k]

    return img_max


def main():
    file_path = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/antennal_lobe_1/'
    img, head = nrrd.read(file_path+'5793_2782_11411-background_subtract.nrrd')
    img_local_max = find_local_max(img)

    img_threshold, head = nrrd.read(file_path+'img_thresholded.nrrd')
    img_local_max = img_local_max * img_threshold
    nrrd.write(file_path+"local_maxima_3d.nrrd", img_local_max)

    img_synapse, head = nrrd.read(file_path+"mask_synapse.nrrd")
    img_synapse[img_synapse!=0] = 1
    local_max_synapse = img_local_max * img_synapse
    nrrd.write(file_path+"local_maxima_synapse_3d.nrrd", local_max_synapse)

    img_junk, head = nrrd.read(file_path+"mask_junk.nrrd")
    img_junk[img_junk!=0] = 1
    local_max_junk = img_local_max * img_junk
    nrrd.write(file_path+"local_maxima_junk_3d.nrrd", local_max_junk)

    print("Done!")
    

if __name__ == "__main__":
    main()
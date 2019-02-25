import nrrd 
import os
import numpy as np 
from skimage.measure import label, regionprops


def generate_individual_mask(img_name, save_path):

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    print("Processing...")
    img, head = nrrd.read(img_name)
    img[img!=0] = 1
    label_img = label(img, neighbors=8)
    regprop_img = regionprops(label_img)

    for i in range(len(regprop_img)):
        curr_region = np.zeros(img.shape, dtype=img.dtype)
        curr_region[label_img==regprop_img[i].label] = 255
        nrrd.write(save_path+str(i)+'.nrrd', curr_region)

    print("Done!")
    return None


def main():
    # file_path1 = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/For_check/'
    # img_name1 = 'watershed_close_prediction_unet.nrrd'
    # generate_individual_mask(file_path1+img_name1, file_path1+'unet_results/')

    file_path2 = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/test2/L2_20180504_neuron0/'
    img_name2 = 'watershed_close_prediction_unet.nrrd'
    generate_individual_mask(file_path2+img_name2, file_path2+'unet_results/')


if __name__ == "__main__":
    main()
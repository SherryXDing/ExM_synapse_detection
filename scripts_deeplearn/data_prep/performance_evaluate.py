from skimage.measure import label, regionprops
import numpy as np 
import nrrd
from skimage import io


def tif_read(file_name):
    """
    read tif image in (rows,cols,slices) shape
    """
    im = io.imread(file_name)
    im_array = np.zeros((im.shape[1],im.shape[2],im.shape[0]), dtype=im.dtype)
    for i in range(im.shape[0]):
        im_array[:,:,i] = im[i]
    return im_array


def calculate_performance(predict_img, ground_truth_img):
    """ Calculate sensitivity and precision
    sensitivity = TP/(TP+FN)
    precision = TP/(TP+FP)
    args:
    if predict_img is prediction, and ground_truth_img is ground truth, then this function calculates precision
    if predict_img is ground truth, and ground_truth_img is prediction, then this function calculates sensitivity
    """
    TP = 0
    FP = 0

    assert predict_img.shape == ground_truth_img.shape, \
        "Prediction does not have the same shape as ground truth!"
    
    predict_img[predict_img!=0] = 1
    label_predict_img = label(predict_img, neighbors=8)
    regionprop_predict_img = regionprops(label_predict_img)

    for i in range(len(regionprop_predict_img)):
        curr_region = np.zeros(predict_img.shape, dtype=predict_img.dtype)
        curr_region[label_predict_img==regionprop_predict_img[i].label] = 1
        curr_obj = curr_region * ground_truth_img
        num_nonzero = np.count_nonzero(curr_obj)
        if num_nonzero > 0:
            TP += 1
        else:
            FP += 1
    
    return TP/(TP+FP)


def main():
    data_path = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/For_check/'
    ground_truth = 'mask_synapse_all.nrrd'
    # vgg_result = 'vgg_final_result.nrrd'
    unet_result = 'unet_new/postprocessed_unet_C2-5753_2694_2876-PLP.tif'

    ground_truth_img, head = nrrd.read(data_path+ground_truth)
    # vgg_img, head = nrrd.read(data_path+vgg_result)
    unet_img = tif_read(data_path+unet_result)

    # precision_vgg = calculate_performance(vgg_img, ground_truth_img)
    # sensitivity_vgg = calculate_performance(ground_truth_img, vgg_img)
    # print("VGG")
    # print("Sensivity = {}".format(sensitivity_vgg))
    # print("Precision = {}".format(precision_vgg))

    precision_unet = calculate_performance(unet_img, ground_truth_img)
    sensitivity_unet = calculate_performance(ground_truth_img, unet_img)
    print("Unet")
    print("Sensitivity = {}".format(sensitivity_unet))
    print("Precision = {}".format(precision_unet))


if __name__ == "__main__":
    main()
from skimage.measure import label, regionprops
import numpy as np 
import nrrd


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
    ground_truth = 'Segmented_mask/mask_synapse_all.nrrd'
    vgg_result = 'final_result.nrrd'
    unet_result = 'prediction_unet.nrrd'
    watershed_unet_result = 'watershed_prediction_unet.nrrd'
    watershed_close_unet_result = 'watershed_close_prediction_unet.nrrd'
    watershed_closefill_unet_result = 'watershed_closefill_prediction_unet.nrrd'

    ground_truth_img, head = nrrd.read(data_path+ground_truth)
    masked_ground_truth_img = np.zeros(ground_truth_img.shape, dtype=ground_truth_img.dtype)
    masked_ground_truth_img[20:-24, 20:-24, 20:-24] = ground_truth_img[20:-24, 20:-24, 20:-24]
    vgg_img, head = nrrd.read(data_path+vgg_result)
    masked_vgg_img = np.zeros(vgg_img.shape, dtype=vgg_img.dtype)
    masked_vgg_img[20:-24, 20:-24, 20:-24] = vgg_img[20:-24, 20:-24, 20:-24]
    unet_img, head = nrrd.read(data_path+unet_result)
    watershed_unet_img, head = nrrd.read(data_path+watershed_unet_result)
    watershed_close_unet_img, head = nrrd.read(data_path+watershed_close_unet_result)
    watershed_closefill_unet_img, head = nrrd.read(data_path+watershed_closefill_unet_result)

    precision_vgg = calculate_performance(masked_vgg_img, masked_ground_truth_img)
    sensitivity_vgg = calculate_performance(masked_ground_truth_img, masked_vgg_img)
    print("VGG")
    print("Sensivity = {}".format(sensitivity_vgg))
    print("Precision = {}".format(precision_vgg))

    precision_unet = calculate_performance(unet_img, masked_ground_truth_img)
    sensitivity_unet = calculate_performance(masked_ground_truth_img, unet_img)
    print("Unet")
    print("Sensitivity = {}".format(sensitivity_unet))
    print("Precision = {}".format(precision_unet))

    precision_watershed_unet = calculate_performance(watershed_unet_img, masked_ground_truth_img)
    sensitivity_watershed_unet = calculate_performance(masked_ground_truth_img, watershed_unet_img)
    print("Unet watershed")
    print("Sensitivity = {}".format(sensitivity_watershed_unet))
    print("Precision = {}".format(precision_watershed_unet))

    precision_watershed_close_unet = calculate_performance(watershed_close_unet_img, masked_ground_truth_img)
    sensitivity_watershed_close_unet = calculate_performance(masked_ground_truth_img, watershed_close_unet_img)
    print("Unet closing watershed")
    print("Sensitivity = {}".format(sensitivity_watershed_close_unet))
    print("Precision = {}".format(precision_watershed_close_unet))

    precision_watershed_closefill_unet = calculate_performance(watershed_closefill_unet_img, masked_ground_truth_img)
    sensitivity_watershed_closefill_unet = calculate_performance(masked_ground_truth_img, watershed_closefill_unet_img)
    print("Unet closing filling watershed")
    print("Sensitivity = {}".format(sensitivity_watershed_closefill_unet))
    print("Precision = {}".format(precision_watershed_closefill_unet))


if __name__ == "__main__":
    main()
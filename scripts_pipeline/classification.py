import nrrd 
import numpy as np 
from skimage.measure import label, regionprops
from keras.models import load_model
from keras import backend as K 
import os 
import math
import time


def _expand_img(img):
    """
    expand 3D image by concatenating the original in three dimensions
    """
    img = np.concatenate((img,img,img), axis=0)
    img = np.concatenate((img,img,img), axis=1)
    img = np.concatenate((img,img,img), axis=2)
    return img


def L1_err(y_true, y_pred):
    score = K.mean(K.abs(y_pred-y_true), axis=-1)
    return score


def classify_local_maxima(local_maxima_img, segment_img, save_path=None):
    '''
    Classify synapses and junks based on number of local maxima points
    Local maxima points > 4 --> synapse; local maxima points < 2 --> junk
    Return an image including synapses with value 255 and undetermined ones with value 1  
    args: 
    local_maxima_img: image including local maxima points
    segment_img: segmented image
    save_path: path to save the detected synapses individually
    '''
    assert local_maxima_img.shape == segment_img.shape, \
        "Segmented image size and local maxima image size are not match!"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print("Classifying based on statistics of local maxima points...")

    segment_img[segment_img!=0] = 1
    label_segment_img = label(segment_img, neighbors=8)
    regprop_segment_img = regionprops(label_segment_img)
    classified_img = np.zeros(segment_img.shape, dtype=segment_img.dtype)

    for i in range(len(regprop_segment_img)):
        curr_region = np.zeros(segment_img.shape, dtype=segment_img.dtype)
        curr_region[label_segment_img==regprop_segment_img[i].label] = 255
        curr_maxima = curr_region * local_maxima_img
        num_local_maxima = np.count_nonzero(curr_maxima)
        if num_local_maxima >4:
            classified_img[label_segment_img==regprop_segment_img[i].label] = 255
            if save_path:
                nrrd.write(save_path+'synapse_stats_'+str(i)+'.nrrd', curr_region)
        elif 2<= num_local_maxima <=4:
            classified_img[label_segment_img==regprop_segment_img[i].label] = 1
        else:
            if save_path:
                nrrd.write(save_path+'junk_stats_'+str(i)+'.nrrd', curr_region)
    
    if save_path:
        nrrd.write(save_path+'stats_classification_result.nrrd', classified_img)

    return classified_img


def classify_obj_using_vgg(classified_img, raw_img, vgg_network, net_input_sz=(48,48,48), custom_objects=None, save_path=None):
    '''
    Classify undetermined objects (2<=number of local maxima<=4) using trained VGG network
    Return a final classification image combining statistical classification results (classified based on number of local maxima)
    and VGG classification results
    args:
    classified_img: an image of statistical classification results (synapse value 255, undetermined objects value 1)
    raw_img: background subtracted raw image
    vgg_network: a trained VGG network in .h5 format (provide with file path)
    custom_objects: any custom loss function/matrices defined in vgg_network
    '''
    assert classified_img.shape == raw_img.shape, \
        "Classified image size and raw image size are not match!"
    assert os.path.exists(vgg_network), \
        "VGG classifier does not exist!"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print("Classifying undetermined objects using VGG network...")

    final_img = np.zeros(classified_img.shape, dtype=classified_img.dtype)
    final_img[classified_img==255] = 255
    
    classified_img[classified_img!=1] = 0  # undetermined objects only
    label_classified_img = label(classified_img, neighbors=8)
    region_props = regionprops(label_classified_img)
    undetermined_obj = np.zeros((len(region_props), net_input_sz[0], net_input_sz[1], net_input_sz[2], 1), dtype=raw_img.dtype)
    region_idx = []  # record object index of region_props in undetermined_obj
    idx = 0

    for i in range(len(region_props)):        
        min_row, min_col, min_slice, max_row, max_col, max_slice = region_props[i].bbox
        center = [min_row+math.floor((max_row-min_row)/2), min_col+math.floor((max_col-min_col)/2), min_slice+math.floor((max_slice-min_slice)/2)]
        # if current object size is bigger than net_input_sz (48,48,48) --> synapse
        if max_row-min_row > net_input_sz[0] or max_col-min_col > net_input_sz[1] or max_slice-min_slice > net_input_sz[2]:
            final_img[label_classified_img==region_props[i].label] = 255
            continue

        curr_obj_mask = np.zeros(raw_img.shape, dtype=raw_img.dtype)
        curr_obj_mask[label_classified_img==region_props[i].label] = 1
        curr_obj = curr_obj_mask * raw_img
        new_curr_obj = _expand_img(curr_obj)
        new_center = [x+y for x,y in zip(center, [raw_img.shape[0],raw_img.shape[1],raw_img.shape[2]])]
        sample_img = new_curr_obj[int(new_center[0]-net_input_sz[0]/2):int(new_center[0]+net_input_sz[0]/2), \
                int(new_center[1]-net_input_sz[1]/2):int(new_center[1]+net_input_sz[1]/2), \
                int(new_center[2]-net_input_sz[2]/2):int(new_center[2]+net_input_sz[2]/2)]
        undetermined_obj[idx,:,:,:,0] = sample_img
        idx += 1
        region_idx.append(i)
    
    undetermined_obj = undetermined_obj[:len(region_idx),:,:,:,:]
    undetermined_obj = np.float32(undetermined_obj)
    # classification using trained VGG
    print("...Loading VGG network...")
    model = load_model(vgg_network, custom_objects=custom_objects)
    prediction = model.predict(undetermined_obj, batch_size=16)

    for i in range(len(region_idx)):
        curr_region_idx = region_idx[i]
        curr_region = np.zeros(classified_img.shape, dtype=classified_img.dtype)
        curr_region[label_classified_img==region_props[curr_region_idx].label] = 255

        if prediction[i] > 0.5:
            final_img[label_classified_img==region_props[curr_region_idx].label] = 255
            if save_path:
                nrrd.write(save_path+'synapse_vgg_'+str(i)+'.nrrd', curr_region)
        else:
            if save_path:
                nrrd.write(save_path+'junk_vgg_'+str(i)+'.nrrd', curr_region)

    if save_path:
        nrrd.write(save_path+'synapse_final_result.nrrd', final_img)

    return final_img


def main():

    data_path = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/test2/L2_20180503_neuron1/'
    raw_img_name = 'BGsubtract_prod_mask_1.nrrd'
    local_maxima_name = 'max_BGsubtract_prod_mask_1.nrrd'
    segment_name = 'watershed_seg_BGsubtract_prod_mask_1.nrrd'
    save_path = data_path+'results/'
    
    local_maxima_img, head = nrrd.read(data_path+local_maxima_name)
    segment_img, head = nrrd.read(data_path+segment_name)
    raw_img, head = nrrd.read(data_path+raw_img_name)

    start = time.time()
    classified_img = classify_local_maxima(local_maxima_img, segment_img, save_path=save_path)
    # nrrd.write(data_path+'stats_'+segment_name, classified_img)
    # classified_img_name = 'stats_watershed_seg_C2_5753_2694_2876.nrrd'
    # classified_img, head = nrrd.read(data_path+classified_img_name)

    vgg_network = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/model_DNN/saved_vgg_model/model_vgg2/vgg2.whole.h5'
    custom_objs = {'L1_err': L1_err}
    final_img = classify_obj_using_vgg(classified_img, raw_img, vgg_network, custom_objects=custom_objs, save_path=save_path)
    # nrrd.write(data_path+'final_result.nrrd', final_img)
    end = time.time()
    print("...Running time of classification is {}".format(end-start))


if __name__ == "__main__":
    main()
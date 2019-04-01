import numpy as np 
import skimage.io
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import gc


def tif_read(file_name):
    """
    read tif image in (rows,cols,slices) shape
    """
    im = skimage.io.imread(file_name)
    im_array = np.zeros((im.shape[1],im.shape[2],im.shape[0]), dtype=im.dtype)
    for i in range(im.shape[0]):
        im_array[:,:,i] = im[i]
    return im_array


def tif_write(im_array, file_name):
    """
    write an array with (rows,cols,slices) shape into a tif image
    """
    im = np.zeros((im_array.shape[2],im_array.shape[0],im_array.shape[1]), dtype=im_array.dtype)
    for i in range(im_array.shape[2]):
        im[i] = im_array[:,:,i]
    skimage.io.imsave(file_name,im)
    return None


def masked_binary_crossentropy(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true,2), K.floatx())
    score = K.mean(K.binary_crossentropy(y_pred*mask, y_true*mask), axis=-1)
    return score


def masked_accuracy(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true,2), K.floatx())
    score = K.mean(K.equal(y_true*mask, K.round(y_pred*mask)), axis=-1)
    return score


def masked_error_pos(y_true, y_pred):
    mask = K.cast(K.equal(y_true,1), K.floatx())
    error = (1-y_pred) * mask
    score = K.sum(error) / K.maximum(K.sum(mask),1)
    return score


def masked_error_neg(y_true, y_pred):
    mask = K.cast(K.equal(y_true,0), K.floatx())
    error = y_pred * mask
    score = K.sum(error) / K.maximum(K.sum(mask),1)
    return score


def unet_test(img, input_sz=(64,64,64), step=(24,24,24), mask=None):
    '''
    Test 3D-Unet on an image data
    args:
    img: image data for testing
    input_sz: U-net input size
    step: number of voxels to move the sliding window in x-,y-,z- direction
    mask: (optional) a mask applied to the img 
    '''
    
    print("Doing prediction using 3D-Unet...")
    model_file = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/model_DNN/saved_unet_model/model_gen2_final/unet.whole.h5'
    unet_model = load_model(model_file, custom_objects={'masked_binary_crossentropy':masked_binary_crossentropy, \
        'masked_accuracy':masked_accuracy, 'masked_error_pos':masked_error_pos, 'masked_error_neg':masked_error_neg})

    gap = (int((input_sz[0]-step[0])/2), int((input_sz[1]-step[1])/2), int((input_sz[2]-step[2])/2))
    img = np.float32(img)
    
    if mask is not None:
        assert mask.shape == img.shape, \
            "Mask and image shapes do not match!"
        mask[mask!=0] = 1
        img = img * mask
        mask = 1-mask
        mask = np.uint8(mask)
        img_masked = np.ma.array(img, mask=mask)
        img = (img - img_masked.mean()) / img_masked.std()
    else:
        img = (img - img.mean()) / img.std()

    # expand the image to deal with edge issue
    new_img = np.zeros((img.shape[0]+gap[0]+input_sz[0], img.shape[1]+gap[1]+input_sz[1], img.shape[2]+gap[2]+input_sz[2]), dtype=img.dtype)
    new_img[gap[0]:new_img.shape[0]-input_sz[0], gap[1]:new_img.shape[1]-input_sz[1], gap[2]:new_img.shape[2]-input_sz[2]] = img
    img = new_img
    predict_img = np.zeros(img.shape, dtype=img.dtype)

    for row in range(0, img.shape[0]-input_sz[0], step[0]):
        for col in range(0, img.shape[1]-input_sz[1], step[1]):
            for vol in range(0, img.shape[2]-input_sz[2], step[2]):
                patch_img = np.zeros((1, input_sz[0], input_sz[1], input_sz[2], 1), dtype=img.dtype)
                patch_img[0,:,:,:,0] = img[row:row+input_sz[0], col:col+input_sz[1], vol:vol+input_sz[2]]
                patch_predict = unet_model.predict(patch_img)
                predict_img[row+gap[0]:row+gap[0]+step[0], col+gap[1]:col+gap[1]+step[1], vol+gap[2]:vol+gap[2]+step[2]] \
                    = patch_predict[0,:,:,:,0]

    predict_img[predict_img>=0.5] = 255
    predict_img[predict_img<0.5] = 0
    predict_img = np.uint8(predict_img)
    predict_img = predict_img[gap[0]:predict_img.shape[0]-input_sz[0], gap[1]:predict_img.shape[1]-input_sz[1], gap[2]:predict_img.shape[2]-input_sz[2]]

    K.clear_session()
    gc.collect()
    print("U-Net DONE!")
    return predict_img
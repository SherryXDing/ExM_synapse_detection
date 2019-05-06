"""
Test the whole pipeline and Conda configurations
"""

import numpy as np 
from exm_deeplearn_lib.exmsyn_models import deepmask_like
from exm_deeplearn_lib.exmsyn_compile import prepare_data, get_file_names
from exm_deeplearn_lib.exmsyn_network import seg_binary_logistic_regression_error, score_binary_logistic_regression_error
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint


# prepare data
pos_path = ['/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/samples/2655-4788-5446/pos/',\
    '/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/samples/4179-2166-3448/pos/']
neg_path = ['/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/samples/2655-4788-5446/neg/', \
    '/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/samples/4179-2166-3448/neg/']

neg_mask_val = False
is_zoom = 0.25
vali_pct = 0.25

print("Get file names of positive samples...")
pos_img_mask_names = get_file_names(pos_path)
print("Prepare data for positive samples...")  
pos_train_img, pos_train_mask, pos_vali_img, pos_vali_mask = prepare_data(pos_img_mask_names, vali_pct, neg_mask_val, is_zoom)
pos_train_score = [np.float32(1)] * pos_train_img.shape[0]
pos_vali_score = [np.float32(1)] * pos_vali_img.shape[0]

print("Get file names of negative samples...")
neg_img_mask_names = get_file_names(neg_path)
print("Prepare data for negative samples...")
neg_train_img, neg_train_mask, neg_vali_img, neg_vali_mask = prepare_data(neg_img_mask_names, vali_pct, neg_mask_val, is_zoom)
neg_train_score = [np.float32(0)] * neg_train_img.shape[0]
neg_vali_score = [np.float32(0)] * neg_vali_img.shape[0]

print("Prepare all traiing samples...")
all_train_img = np.concatenate((pos_train_img,neg_train_img), axis=0)
all_vali_img = np.concatenate((pos_vali_img,neg_vali_img), axis=0)
all_train_mask = np.concatenate((pos_train_mask,neg_train_mask), axis=0)
print("Prepare all validation samples...")
all_vali_mask =  np.concatenate((pos_vali_mask,neg_vali_mask), axis=0)
all_train_score = np.concatenate((pos_train_score,neg_train_score), axis=0)
all_vali_score = np.concatenate((pos_vali_score,neg_vali_score), axis=0)

print("Concatenate training samples and validation samples...")
all_train_img = np.concatenate((all_train_img,all_vali_img), axis=0)  # concatenate validation data at the end of training data
all_train_mask = np.concatenate((all_train_mask,all_vali_mask), axis=0)
all_train_score = np.concatenate((all_train_score,all_vali_score), axis=0)


# prepare model
input_shape = (56,56,48)
model = deepmask_like(input_shape)
sgd_opti = SGD(lr=0.001, momentum=0.9, decay=0.00005, nesterov=True)
model.compile(optimizer=sgd_opti, loss={'seg_out':seg_binary_logistic_regression_error, 'score_out':score_binary_logistic_regression_error}, metrics=['accuracy'])
model.summary()

save_model_path = "/groups/scicompsoft/home/dingx/Documents/ExM/scripts_deeplearn/saved_model/weights.{epoch:02d}.hdf5"
check_point = ModelCheckpoint(filepath=save_model_path, monitor='val_loss', verbose=1, save_best_only=True)
history = model.fit({'in':all_train_img}, {'seg_out':all_train_mask, 'score_out':all_train_score}, batch_size=32, \
        epochs=1, callbacks=[check_point], validation_split=vali_pct, shuffle=True)
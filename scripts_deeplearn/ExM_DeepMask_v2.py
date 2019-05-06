"""
Test data generator and multi-gpu training
"""

from exm_deeplearn_lib.exmsyn_models import deepmask_like
from exm_deeplearn_lib.exmsyn_compile import get_file_names, gen_deepmask_batch, prepare_validation_set
from exm_deeplearn_lib.exmsyn_network import seg_binary_logistic_regression_error, score_binary_logistic_regression_error
from keras.optimizers import SGD
import numpy as np
from keras.utils import multi_gpu_model


# prepare data
pos_path = ['/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/samples/2655-4788-5446/pos/', \
    '/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/samples/4179-2166-3448/pos/']
neg_path = ['/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/samples/2655-4788-5446/neg/', \
    '/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/samples/4179-2166-3448/neg/']

print("Get file names of positive samples...")
pos_img_mask_names = get_file_names(pos_path)
pos_labels = [1] * len(pos_img_mask_names)
print("Get file names of negative samples...")
neg_img_mask_names = get_file_names(neg_path)
neg_labels = [0] * len(neg_img_mask_names)

vali_pct = 0.1  # percentage of data as validation 
num_pos_vali = np.int(np.round(len(pos_img_mask_names)*vali_pct))
num_neg_vali = np.int(np.round(len(neg_img_mask_names)*vali_pct))

img_mask_names = pos_img_mask_names[num_pos_vali:] + neg_img_mask_names[num_neg_vali:]
labels = pos_labels[num_pos_vali:] + neg_labels[num_neg_vali:]
vali_names = pos_img_mask_names[:num_pos_vali] + neg_img_mask_names[:num_neg_vali]
vali_labels = pos_labels[:num_pos_vali] + neg_labels[:num_neg_vali]

# prepare and train model
input_shape = (56,56,48)
n_gpus = 2
model = deepmask_like(input_shape)
sgd_opti = SGD(lr=0.001, momentum=0.9, decay=0.00005, nesterov=True)
model.compile(optimizer=sgd_opti, loss={'seg_out':seg_binary_logistic_regression_error, 'score_out':score_binary_logistic_regression_error}, metrics=['accuracy'])
model.summary()
parallel_model = multi_gpu_model(model=model,gpus=n_gpus)
parallel_model.compile(optimizer=sgd_opti, loss={'seg_out':seg_binary_logistic_regression_error, 'score_out':score_binary_logistic_regression_error}, metrics=['accuracy'])

neg_mask_val = False
is_zoom = 0.25  # zoom mask to 1/4 of the original
batch_sz = 32
history = parallel_model.fit_generator(gen_deepmask_batch(img_mask_names,labels,batch_sz*n_gpus,neg_mask_val,is_zoom), steps_per_epoch=500, epochs=1)

# validate model
vali_img, vali_mask = prepare_validation_set(vali_names, neg_mask_val, is_zoom)
vali_labels = np.asarray(vali_labels)
score = model.evaluate(x=vali_img, y=[vali_mask, vali_labels])
out_mask, out_label = model.predict(x=vali_img)
from exm_deeplearn_lib.exmsyn_models import deepmask_like
from exm_deeplearn_lib.exmsyn_compile import get_file_names, gen_deepmask_batch_general
from exm_deeplearn_lib.exmsyn_network import seg_binary_logistic_regression_error, score_binary_logistic_regression_error
from exm_deeplearn_lib.exmsyn_network import DeepNeuralNetwork
from keras.optimizers import SGD
import numpy as np
from random import shuffle
import pickle 


# prepare data
pos_path = ['/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/antennal_lobe_1/samples/pos/', \
    '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/ellipsoid_body_1/samples/pos/', \
    '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/mushroom_body_1/samples/pos/', \
    '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/optic_lobe_1/samples/pos/', \
    '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/protocerebrum_1/samples/pos/']
neg_path = ['/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/antennal_lobe_1/samples/neg/', \
    '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/ellipsoid_body_1/samples/neg/', \
    '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/mushroom_body_1/samples/neg/', \
    '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/optic_lobe_1/samples/neg/', \
    '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/protocerebrum_1/samples/neg/']

print("Get file names of positive samples...")
pos_img_mask_names = get_file_names(pos_path)
shuffle(pos_img_mask_names)
print("Get file names of negative samples...")
neg_img_mask_names = get_file_names(neg_path)
shuffle(neg_img_mask_names)

vali_pct = 0.1
num_pos_vali = np.int(np.round(len(pos_img_mask_names)*vali_pct))
num_neg_vali = np.int(np.round(len(neg_img_mask_names)*vali_pct))

train_pos_img_mask_names = pos_img_mask_names[num_pos_vali:]
train_neg_img_mask_names = neg_img_mask_names[num_neg_vali:]
vali_img_mask_names = [pos_img_mask_names[:num_pos_vali], neg_img_mask_names[:num_neg_vali]]  # save for later use
saved_folder = '/groups/scicompsoft/home/dingx/Documents/ExM/scripts_deeplearn/saved_deepmask_model/'
# with open(saved_folder+'vali_sample.pkl','wb') as f:
#     pickle.dump(vali_img_mask_names, f)

input_shape = (64,64,64)
model = deepmask_like(input_shape)
sgd_opti = SGD(lr=0.001, momentum=0.9, decay=0.00005, nesterov=True)
# loss_func={'seg_out':seg_binary_logistic_regression_error, 'score_out':score_binary_logistic_regression_error}
loss_func={'seg_out': 'binary_crossentropy', 'score_out': 'binary_crossentropy'}
compile_args = {'optimizer':sgd_opti, 'loss':loss_func, 'metrics':['accuracy']}
network = DeepNeuralNetwork(model, compile_args=compile_args)

batch_sz = 16
n_gpus = 4
generator = gen_deepmask_batch_general(train_pos_img_mask_names, train_neg_img_mask_names, batch_sz=batch_sz*n_gpus, is_zoom=0.25)
history = network.train_network(generator=generator, steps_per_epoch=100, epochs=500, n_gpus=n_gpus, save_name=None)

with open(saved_folder+'history_batch64_steps100_epochs500.pkl', 'wb') as f:
    pickle.dump(history.history, f)
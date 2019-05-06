from exm_deeplearn_lib.exmsyn_models import vgg_like_v3
from exm_deeplearn_lib.exmsyn_compile import gen_vgg_batch
from exm_deeplearn_lib.exmsyn_network import DeepNeuralNetwork
from exm_deeplearn_lib.exmsyn_network import L1_err
from exm_deeplearn_lib.exmsyn_compile import prepare_vgg_validation_set
from keras.optimizers import SGD
import pickle
import os
from random import shuffle
import numpy as np 

'''Use vgg_v3'''

# # prepare data files
# pos_path = ['/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/antennal_lobe_1/samples_overlapped_maxima/pos/', \
#     '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/ellipsoid_body_1/samples_overlapped_maxima/pos/', \
#     '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/mushroom_body_1/samples_overlapped_maxima/pos/', \
#     '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/optic_lobe_1/samples_overlapped_maxima/pos/', \
#     '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/protocerebrum_1/samples_overlapped_maxima/pos/']
# neg_path = ['/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/antennal_lobe_1/samples_overlapped_maxima/neg/', \
#     '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/ellipsoid_body_1/samples_overlapped_maxima/neg/', \
#     '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/mushroom_body_1/samples_overlapped_maxima/neg/', \
#     '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/optic_lobe_1/samples_overlapped_maxima/neg/', \
#     '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/protocerebrum_1/samples_overlapped_maxima/neg/']

# pos_img_names = []
# for i in range(len(pos_path)):
#     for name in os.listdir(pos_path[i]):
#         curr_name = pos_path[i]+name
#         pos_img_names.append(curr_name)
# shuffle(pos_img_names)

# neg_img_names = []
# for i in range(len(neg_path)):
#     for name in os.listdir(neg_path[i]):
#         curr_name = neg_path[i]+name
#         neg_img_names.append(curr_name)
# shuffle(neg_img_names)

# # separate training and validation files and save to disk
# vali_pct = 0.1
# num_pos_vali = np.int(np.round(len(pos_img_names)*vali_pct))
# num_neg_vali = np.int(np.round(len(neg_img_names)*vali_pct))

# train_pos_img_names = pos_img_names[num_pos_vali:]
# train_neg_img_names = neg_img_names[num_neg_vali:]
# train_img_names = [train_pos_img_names, train_neg_img_names]
# test_pos_img_names = pos_img_names[:num_pos_vali]
# test_neg_img_names = neg_img_names[:num_neg_vali]
# test_img_names = [test_pos_img_names, test_neg_img_names]
# save_path = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/model_DNN/saved_vgg_model/'
# with open(save_path+'train_sample.pkl', 'wb') as f:
#     pickle.dump(train_img_names, f)
# with open(save_path+'test_sample.pkl', 'wb') as f:
#     pickle.dump(test_img_names, f)


save_path = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/model_DNN/saved_vgg_model/'
with open(save_path+'train_sample.pkl', 'rb') as f:
    train_img_names = pickle.load(f)
train_pos_img_names = train_img_names[0]
train_neg_img_names = train_img_names[1]

with open(save_path+'test_sample.pkl', 'rb') as f:
    test_img_names = pickle.load(f)
test_pos_img_names = test_img_names[0]
test_neg_img_names = test_img_names[1]


# validation set
test_img, test_label = prepare_vgg_validation_set(test_pos_img_names, test_neg_img_names)

# build VGG
input_shape = (48,48,48)
model = vgg_like_v3(input_shape)
sgd_opti = SGD(lr=0.001, momentum=0.9, decay=0.00005, nesterov=True)
compile_args = {'optimizer':sgd_opti, 'loss':'binary_crossentropy', 'metrics':['accuracy', L1_err]}
network = DeepNeuralNetwork(model, compile_args=compile_args)

batch_sz = 16
n_gpus = 4
generator = gen_vgg_batch(train_pos_img_names, train_neg_img_names, batch_sz=batch_sz*n_gpus)
history = network.train_network(generator=generator, steps_per_epoch=100, epochs=2000, n_gpus=n_gpus, save_name=None, validation_data=(test_img, test_label))

with open(save_path+'history_lr1e-3_sgd_batch64_steps100_epochs2000_L1err_vgg3.pkl', 'wb') as f:
    pickle.dump(history.history, f)
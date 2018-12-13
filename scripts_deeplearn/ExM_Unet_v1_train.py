from exm_deeplearn_lib.exmsyn_models import unet_like
from exm_deeplearn_lib.exmsyn_compile import gen_unet_batch_v1
from exm_deeplearn_lib.exmsyn_network import masked_binary_crossentropy
from exm_deeplearn_lib.exmsyn_network import masked_accuracy, masked_error_pos, masked_error_neg
from exm_deeplearn_lib.exmsyn_network import DeepNeuralNetwork
from keras.optimizers import SGD
import pickle


file_path = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/'
img_mask_names = [(file_path+'antennal_lobe_1/5793_2782_11411.nrrd', file_path+'antennal_lobe_1/mask_unet_syn1_edge2.nrrd'), \
    (file_path+'ellipsoid_body_1/C1-7527_3917_6681.nrrd',file_path+'ellipsoid_body_1/mask_unet_syn1_edge2.nrrd'), \
    (file_path+'mushroom_body_1/C1-6210_6934_5492.nrrd', file_path+'mushroom_body_1/mask_unet_syn1_edge2.nrrd'), \
    (file_path+'optic_lobe_1/C1-4228_3823_4701.nrrd', file_path+'optic_lobe_1/mask_unet_syn1_edge2.nrrd'), \
    (file_path+'protocerebrum_1/C1-13904_10064_4442.nrrd', file_path+'protocerebrum_1/mask_unet_syn1_edge2.nrrd')]

input_shape = (64,64,64)
model = unet_like(input_shape)
sgd_opti = SGD(lr=0.001, momentum=0.9, decay=0.00005, nesterov=True)
compile_args = {'optimizer':sgd_opti, 'loss':masked_binary_crossentropy, 'metrics':[masked_accuracy, masked_error_pos, masked_error_neg]}
network = DeepNeuralNetwork(model, compile_args=compile_args)

batch_sz = 16
n_gpus = 4
generator = gen_unet_batch_v1(img_mask_names, crop_sz=(64,64,64), mask_sz=(24,24,24), batch_sz=batch_sz*n_gpus)
save_path = '/groups/scicompsoft/home/dingx/Documents/ExM/scripts_deeplearn/saved_unet_model/'
history = network.train_network(generator=generator, steps_per_epoch=100, epochs=500, n_gpus=n_gpus, save_name=None)

with open(save_path+'history_rawdata_batch64_steps100_epochs500.pkl', 'wb') as f:
    pickle.dump(history.history, f)
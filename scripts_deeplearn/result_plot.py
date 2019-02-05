import pickle
import matplotlib.pyplot as plt 
import numpy as np 

history_path = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/model_DNN/saved_unet_model/model_gen2/'
file_name = 'history_rawdata_gen2_lr1e-3_sgd_batch64_steps100_epochs2000.pkl'


with open(history_path+file_name, 'rb') as f:
    history = pickle.load(f)


# VGG
# num_epoch = len(history['loss'])
# label=['Train_Acc', 'Train_Loss', 'Train_L1_err', 'Vali_acc', 'Vali_L1_err']
# plt.figure()
# plt.plot(range(num_epoch), history['acc'], 'r')
# plt.plot(range(num_epoch), history['loss'], 'b')
# plt.plot(range(num_epoch), history['L1_err'], 'k')
# plt.plot(range(num_epoch), history['val_acc'], 'c')
# # plt.plot(range(num_epoch), history['val_loss'], 'g')
# plt.plot(range(num_epoch), history['val_L1_err'], 'm')
# plt.xlabel('Num epoch')
# plt.ylabel('Performance')
# plt.legend(label)
# plt.grid(color='k', linestyle=':')
# plt.savefig(history_path+'performance_lr1e-3_sgd_batch64_steps100_epochs2000_L1err_vgg2.pdf')
# plt.show()


# U-Net
num_epoch = len(history['loss'])
label=['Acc', 'Loss', 'Err_pos', 'Err_neg']
plt.figure()
plt.plot(range(num_epoch), history['masked_accuracy'], 'r')
plt.plot(range(num_epoch), history['loss'], 'b')
plt.plot(range(num_epoch), history['masked_error_pos'], 'c')
plt.plot(range(num_epoch), history['masked_error_neg'], 'g')
plt.xlabel('Num epoch')
plt.ylabel('Performance')
plt.legend(label)
plt.grid(color='k', linestyle=':')
plt.savefig(history_path+'performance_rawdata_gen2_lr1e-3_sgd_batch64_steps100_epochs2000.pdf')
plt.show()
import pickle
import matplotlib.pyplot as plt 

history_path = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/model_DNN/saved_vgg_model/'
file_name = 'history_batch64_steps100_epochs500_train_vali.pkl'


with open(history_path+file_name, 'rb') as f:
    history = pickle.load(f)


num_epoch = len(history['loss'])
label=['Train_Acc', 'Train_Loss', 'Vali_acc']
plt.figure()
plt.plot(range(num_epoch), history['acc'], 'r')
plt.plot(range(num_epoch), history['loss'], 'b')
plt.plot(range(num_epoch), history['val_acc'], 'c')
#plt.plot(range(num_epoch), history['val_loss'], 'g')
plt.xlabel('Num epoch')
plt.ylabel('Performance')
plt.legend(label)
plt.grid(color='k', linestyle=':')
plt.savefig(history_path+'performance_batch64_steps100_epoch500_train_vali.pdf')
plt.show()

#num_epoch = len(history['loss'])
#label=['Acc', 'Loss', 'Err_pos', 'Err_neg']
#plt.figure()
#plt.plot(range(num_epoch), history['masked_accuracy'], 'r')
#plt.plot(range(num_epoch), history['loss'], 'b')
#plt.plot(range(num_epoch), history['masked_error_pos'], 'c')
#plt.plot(range(num_epoch), history['masked_error_neg'], 'g')
#plt.xlabel('Num epoch')
#plt.ylabel('Performance')
#plt.legend(label)
#plt.grid(color='k', linestyle=':')
#plt.savefig(history_path+'performance_rawdata_gen2_batch64_steps100_epochs500.pdf')
#plt.show()
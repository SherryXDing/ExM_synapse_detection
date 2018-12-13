import pickle
import numpy as np 
from random import shuffle
import nrrd 
import matplotlib.pyplot as plt 
from keras.models import load_model
from keras import backend as K 
import gc


model_path = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/model_DNN/saved_vgg_model/'
vali_data_file = 'test_sample.pkl'
with open(model_path+vali_data_file, 'rb') as f:
    vali_img_names = pickle.load(f)

vali_names = vali_img_names[0] + vali_img_names[1]
vali_labels = [1]*len(vali_img_names[0]) + [0]*len(vali_img_names[1])
combined_names_labels = list(zip(vali_names, vali_labels))
shuffle(combined_names_labels)
vali_names, vali_labels = zip(*combined_names_labels)
vali_names = list(vali_names)
vali_labels = list(vali_labels)


curr_img, head = nrrd.read(vali_names[0])
vali_img = np.zeros((len(vali_names), curr_img.shape[0], curr_img.shape[1], curr_img.shape[2], 1), dtype=curr_img.dtype)
for i in range(len(vali_names)):
    curr_img, head = nrrd.read(vali_names[i])
    vali_img[i,:,:,:,0] = curr_img
vali_img = np.float32(vali_img)
vali_labels = np.asarray(vali_labels)


N_model = 500
loss_all = np.zeros((N_model,1))
acc_all = np.zeros((N_model,1))

for i in range(N_model):
    model = load_model(model_path+'vgg_'+str(i)+'.h5')
    score = model.evaluate(vali_img, vali_labels, batch_size=16)
    loss_all[i] = score[0]
    acc_all[i] = score[1]
    K.clear_session()
    gc.collect()


plt.figure()
plt.plot(range(N_model), acc_all, 'r')
plt.plot(range(N_model), loss_all, 'b')
plt.xlabel('Num epoch')
plt.ylabel('Performance')
plt.legend(['Acc', 'Loss'])
plt.grid(color='k', linestyle=':')
plt.savefig(model_path+'vgg_batch64_steps100_epochs500.pdf', bbox_inches='tight')
plt.show()
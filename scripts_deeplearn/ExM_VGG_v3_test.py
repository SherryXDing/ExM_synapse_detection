import pickle
import numpy as np 
from exm_deeplearn_lib.exmsyn_compile import prepare_vgg_validation_set
import matplotlib.pyplot as plt 
from keras.models import load_model
from keras.models import model_from_json 
from keras.optimizers import SGD
from keras import backend as K 
import gc
from exm_deeplearn_lib.exmsyn_network import L1_err


model_path = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/model_DNN/saved_vgg_model/'
vali_data_file = 'test_sample.pkl'
with open(model_path+vali_data_file, 'rb') as f:
    vali_img_names = pickle.load(f)

vali_img, vali_labels = prepare_vgg_validation_set(vali_img_names[0], vali_img_names[1])


# test saved models after each epoch
N_model = 50
loss_all = np.zeros((N_model,1))
acc_all = np.zeros((N_model,1))
L1_all = np.zeros((N_model,1))

for i in range(N_model):
    model_epoch = load_model(model_path+'model_vgg2/vgg2_'+str(i)+'.h5', custom_objects={'L1_err': L1_err})
    score = model_epoch.evaluate(vali_img, vali_labels, batch_size=16)
    print([score[0], score[1], score[2]])
    loss_all[i] = score[0]
    acc_all[i] = score[1]
    L1_all[i] = score[2]
    K.clear_session()
    gc.collect()

plt.figure()
plt.plot(range(N_model), acc_all, 'r')
# plt.plot(range(N_model), loss_all, 'b')
plt.plot(range(N_model), L1_all, 'g')
plt.xlabel('Num epoch')
plt.ylabel('Performance')
plt.legend(['Acc', 'L1_err'])
plt.grid(color='k', linestyle=':')
plt.savefig(model_path+'model_vgg2/vgg2_batch64_steps100_epochs200.pdf', bbox_inches='tight')
plt.show()


# test saved whole model after all epochs
model_whole = load_model(model_path+'model_vgg2/vgg2.whole.h5', custom_objects={'L1_err':L1_err})
score = model_whole.evaluate(vali_img, vali_labels, batch_size=16)
print([score[0], score[1], score[2]])
prediction = model_whole.predict(vali_img, batch_size=16)


# test saved template model after all epochs
with open(model_path+'model_vgg2/vgg2_arch.json','r') as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights(model_path+'model_vgg2/vgg2_weight.h5')
sgd_opti = SGD(lr=0.001, momentum=0.9, decay=0.00005, nesterov=True)
compile_args = {'optimizer':sgd_opti, 'loss':'binary_crossentropy', 'metrics':['accuracy', L1_err]}
model.compile(**compile_args)
score = model.evaluate(vali_img, vali_labels, batch_size=16)
print([score[0], score[1], score[2]])
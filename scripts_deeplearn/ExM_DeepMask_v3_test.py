import pickle
import numpy as np 
from keras.models import load_model
from exm_deeplearn_lib.exmsyn_compile import prepare_validation_set
from exm_deeplearn_lib.exmsyn_network import seg_binary_logistic_regression_error, score_binary_logistic_regression_error
from exm_deeplearn_lib.exmsyn_network import calculate_iou
from sklearn.metrics import accuracy_score
# from random import shuffle
import matplotlib.pyplot as plt
import gc 
from keras import backend as K 


model_path = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/model_DNN/deepmask_batch64_step50_epoch100_sgd_lr1e-5/'
vali_data_file = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/model_DNN/deepmask_batch64_step50_epoch100_sgd_lr1e-5/vali_sample.pkl'
with open(vali_data_file, 'rb') as f:
    vali_img_mask_names = pickle.load(f)  # vali_img_mask_names[0]: positive samples, vali_img_mask_names[1]: negative samples

# combined_names_labels = list(zip(vali_img_mask_names[0]+vali_img_mask_names[1], vali_lables))
# shuffle(combined_names_labels)
# vali_img_mask_names, vali_label = zip(*combined_names_labels)
# vali_names = list(vali_names)
# vali_label = list(vali_label)
vali_names = vali_img_mask_names[0] + vali_img_mask_names[1]
vali_img, vali_mask = prepare_validation_set(vali_names, is_zoom=0.25)
vali_label = [1]*len(vali_img_mask_names[0]) + [0]*len(vali_img_mask_names[1])
vali_label = np.asarray(vali_label)

N_model = 22
total_loss_all = np.zeros((N_model,1))
seg_loss_all = np.zeros((N_model,1))
score_loss_all = np.zeros((N_model,1))
seg_iou_all = np.zeros((N_model,1))
score_acc_all = np.zeros((N_model,1))

for i in range(N_model):
    model = load_model(model_path+'deepmask_'+str(i)+'.h5', \
        custom_objects={'seg_binary_logistic_regression_error':seg_binary_logistic_regression_error, 'score_binary_logistic_regression_error':score_binary_logistic_regression_error})

    score = model.evaluate(vali_img, [vali_mask, vali_label], batch_size=16)
    total_loss_all[i] = score[0]
    seg_loss_all[i] = score[1]
    score_loss_all[i] = score[2]
    out_mask, out_label = model.predict(vali_img, batch_size=16)
    out_mask[out_mask>0] = 1
    out_mask[out_mask<=0] = 0
    out_label[out_label>0] = 1
    out_label[out_label<=0] = 0
    iou = calculate_iou(vali_mask, out_mask)
    seg_iou_all[i] = iou.mean()
    score_acc_all[i] = accuracy_score(vali_label, out_label)
    K.clear_session()
    gc.collect()


label = ['Total loss', 'Seg loss', 'Score loss', 'IoU', 'Acc']
plt.figure()
plt.plot(range(N_model), total_loss_all, 'b')
plt.plot(range(N_model), seg_loss_all, 'g')
plt.plot(range(N_model), score_loss_all, 'c')
plt.plot(range(N_model), seg_iou_all, 'r')
plt.plot(range(N_model), score_acc_all, 'k')
plt.xlabel('Num epoch')
plt.legend(label)
plt.grid(b=True)
# plt.savefig('deepmask_batch64_step50_lr1e-3.pdf', bbox_inches='tight')
plt.show() 
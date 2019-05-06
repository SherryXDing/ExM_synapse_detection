from keras.models import model_from_json
from keras.utils import multi_gpu_model
import numpy as np
from sklearn.metrics import accuracy_score
from keras.callbacks import Callback, CSVLogger  
from keras import backend as K
import tensorflow as tf 


def score_binary_logistic_regression_error(y_true, y_pred):
    """
    loss function for object score in DeepMask
    """
    score = 1.0/32 * K.log(1 + K.exp(-y_true*y_pred))
    return score


def seg_binary_logistic_regression_error(y_true, y_pred):
    """
    loss function for object mask in DeepMask
    """
    score = K.mean(K.log(1 + K.exp(-y_true*y_pred)), axis=-1)
    return score


def masked_binary_crossentropy(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true,2), K.floatx())
    score = K.mean(K.binary_crossentropy(y_pred*mask, y_true*mask), axis=-1)
    return score


def masked_accuracy(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true,2), K.floatx())
    score = K.mean(K.equal(y_true*mask, K.round(y_pred*mask)), axis=-1)
    return score


def masked_error_pos(y_true, y_pred):
    mask = K.cast(K.equal(y_true,1), K.floatx())
    error = (1-y_pred) * mask
    score = K.sum(error) / K.maximum(K.sum(mask),1)
    return score


def masked_error_neg(y_true, y_pred):
    mask = K.cast(K.equal(y_true,0), K.floatx())
    error = y_pred * mask
    score = K.sum(error) / K.maximum(K.sum(mask),1)
    return score


def L1_err(y_true, y_pred):
    score = K.mean(K.abs(y_pred-y_true), axis=-1)
    return score


def load_architecture(filename):
    json_file = open(filename+'.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(filename+'.h5')
    return model


def calculate_iou(y_true, y_test):
    """
    calcuate IoU = TP/(TP+FN+FP) of segmentation result
    y_true, y_test: arrays of true segmentation and predicted segmentation in shape of (samples, rows, cols, slices), values are 1/0 or 1/-1
    """
    assert y_true.shape == y_test.shape, \
        "Ground truth data and prediction result are in different shape!"

    max_val = np.amax(y_true)  # should be 1 in most cases
    min_val = np.amin(y_true)  # is 0 or -1 depends on neg_mask_val is False or True
    
    IoU = np.zeros((y_true.shape[0],1))
    for i in range(y_true.shape[0]):
        curr_true = y_true[i].flatten()
        curr_test = y_test[i].flatten()
        TP = 0
        FP = 0
        FN = 0
        for j in range(len(curr_true)):
            if curr_true[j]==max_val and curr_test[j]==curr_true[j]:
                TP += 1
            if curr_true[j]==max_val and curr_test[j]!=curr_true[j]:
                FN += 1
            if curr_true[j]==min_val and curr_test[j]!=curr_true[j]:
                FP += 1
        IoU[i] = TP / (TP+FP+FN)
    
    return IoU


class multi_gpu_callback(Callback):
    """
    set callbacks for multi-gpu training
    """
    def __init__(self, model, save_name):
        super().__init__()
        self.model_to_save = model
        self.save_name = save_name
    
    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save("{}_{}.h5".format(self.save_name, epoch))


class DeepNeuralNetwork:
    """
    deep nerual network class that wraps keras model (multi-gpu model) and related functions
    """
    def __init__(self, model, compile_args=None):
        with tf.device('/cpu:0'):
            self.network = model
        if compile_args is None:
            compile_args = {'optimizer':'adam', 'loss':'binary_crossentropy', 'metrics':['accuracy']}
        self.network.compile(**compile_args)
        self.compile_args = compile_args
        self.network.summary()


    def save_whole_network(self, file_path_name):
        file_name = file_path_name + '.whole.h5'
        self.network.save(file_name, overwrite=True)
        print('Save the whole network to disk as a .whole.h5 file.')


    def save_architecture(self, file_path_name):
        model_json = self.network.to_json()
        with open(file_path_name+'_arch.json', 'w') as json_file:
            json_file.write(model_json)
        self.network.save_weights(file_path_name+'_weight.h5', overwrite=True)
        print('Saved network architecture to disk with architecture in .json file and weights in .h5 file.')

    
    def train_network(self, generator, steps_per_epoch=100, epochs=100, n_gpus=1, save_name=None, validation_data=None):
        # save_name: name of saved log file and model after each epoch (result in {save_name}.log and {save_name}_{epoch}.h5 file) 
        if save_name:
            csv_logger = CSVLogger(save_name+'.log')
            check_point = multi_gpu_callback(self.network, save_name)
            callbacks = [csv_logger, check_point]
        else:
            callbacks = None

        if n_gpus == 1:
            print("Training using a single GPU...")
            history = self.network.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks, validation_data=validation_data)
        else:
            print("Training using multiple GPUs...")
            parallel_model = multi_gpu_model(self.network, gpus=n_gpus, cpu_merge=True, cpu_relocation=False)
            parallel_model.compile(**self.compile_args)
            history = parallel_model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks, validation_data=validation_data)
        return history
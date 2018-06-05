# Authors:
# Akshay Chaudhari and Zhongnan Fang
# May 2018
# akshaysc@stanford.edu

from __future__ import print_function, division

import numpy as np
import pickle
import json
import math
import os

from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, add, Lambda, Dropout, AlphaDropout
from keras.layers import BatchNormalization as BN
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K
import keras.callbacks as kc  

from keras.callbacks import ModelCheckpoint, History
from keras.callbacks import LambdaCallback as lcb
from keras.callbacks import LearningRateScheduler as lrs
from keras.callbacks import TensorBoard as tfb

from utils.generator_msk_seg import calc_generator_info, img_generator_oai
from utils.models import unet_2d_model
from utils.losses import dice_loss

# Training and validation data locations
train_path = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/train_aug/'
valid_path = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/valid/'
test_path  = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/test'
train_batch_size = 35
valid_batch_size = 35

# Locations and names for saving training checkpoints
cp_save_path = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/weights'
cp_save_tag = 'unet_2d_men'
pik_save_path = './checkpoint/' + cp_save_tag + '.dat'

# Model parameters
n_epochs = 20
file_types = ['im']
# Tissues are in the following order
# 0. Femoral 1. Lat Tib 2. Med Tib. 3. Pat 4. Lat Men 5. Med Men
tissue = np.arange(0,1)
# Load pre-trained model
model_weights = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/weights/unet_2d_men_weights.009--0.7682.h5'

# training and validation image size
img_size = (288,288,len(file_types))
# What dataset are we training on? 'dess' or 'oai'
tag = 'oai_aug'

# Restrict number of files learned. Default is all []
learn_files = []
# Freeze layers in transfer learning
layers_to_freeze = []

# learning rate schedule
# Implementing a step decay for now
def step_decay(epoch):
    initial_lrate = 1e-4
    drop = 0.8
    epochs_drop = 1.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def train_seg(img_size, train_path, valid_path, train_batch_size, valid_batch_size, 
                cp_save_path, cp_save_tag, n_epochs, file_types, pik_save_path, 
                tag, tissue, learn_files, layers_to_freeze):

    # set image format to be (N, dim1, dim2, dim3, ch)
    K.set_image_data_format('channels_last')
    train_files, train_nbatches = calc_generator_info(train_path, train_batch_size, learn_files)
    valid_files, valid_nbatches = calc_generator_info(valid_path, valid_batch_size)

    # Print some useful debugging information
    print('INFO: Train size: %d, batch size: %d' % (len(train_files), train_batch_size))
    print('INFO: Valid size: %d, batch size: %d' % (len(valid_files), valid_batch_size))    
    print('INFO: Image size: %s' % (img_size,))
    print('INFO: Image types included in training: %s' % (file_types,))    
    print('INFO: Number of tissues being segmented: %d' % len(tissue))
    print('INFO: Number of frozen layers: %s' % len(layers_to_freeze))

    # create the unet model
    model = unet_2d_model(img_size)
    if model_weights is not None:
        model.load_weights(model_weights,by_name=True)

    # Set up the optimizer 
    model.compile(optimizer=Adam(lr=1e-9, beta_1=0.99, beta_2=0.995, epsilon=1e-08, decay=0.0), 
                  loss=dice_loss)

    # Optinal, but this allows you to freeze layers if you want for transfer learning
    for lyr in layers_to_freeze:
        model.layers[lyr].trainable = False

    # model callbacks per epoch
    cp_cb   = ModelCheckpoint(cp_save_path + '/' + cp_save_tag + '_weights.{epoch:03d}-{val_loss:.4f}.h5',save_best_only=True)
    tfb_cb  = tfb('./tf_log',
                  histogram_freq=1,
                  write_grads=False,
                  write_images=False)
    lr_cb   = lrs(step_decay)
    hist_cb = LossHistory()

    callbacks_list = [tfb_cb, cp_cb, hist_cb, lr_cb]

    # Start the training    
    model.fit_generator(
            img_generator_oai(train_path, train_batch_size, img_size, tissue, tag),
            train_nbatches,
            epochs=n_epochs,
            validation_data=img_generator_oai(valid_path, valid_batch_size, img_size, tissue, tag),
            validation_steps=valid_nbatches,
            callbacks=callbacks_list)

    # Save files to write as output
    data = [hist_cb.epoch, hist_cb.lr, hist_cb.losses, hist_cb.val_losses]
    with open(pik_save_path, "wb") as f:
        pickle.dump(data, f)

    return hist_cb


# Print and asve the training history
class LossHistory(kc.Callback):
    def on_train_begin(self, logs={}):
       self.val_losses = []
       self.losses = []
       self.lr = []
       self.epoch = []
 
    def on_epoch_end(self, batch, logs={}):
       self.val_losses.append(logs.get('val_loss'))
       self.losses.append(logs.get('loss'))
       self.lr.append(step_decay(len(self.losses)))
       self.epoch.append(len(self.losses))

if __name__ == '__main__':
  
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    model = unet_2d_model(img_size)
    # print(model.summary())
    train_seg(img_size, train_path, valid_path, train_batch_size, valid_batch_size, 
                cp_save_path, cp_save_tag, n_epochs, file_types, pik_save_path, 
                tag, tissue, learn_files, layers_to_freeze)


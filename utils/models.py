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
from keras.utils import plot_model
from keras import backend as K
import tensorflow as tf

def unet_2d_model(input_size):

    # input size is a tuple of the size of the image
    # assuming channel last
    # input_size = (dim1, dim2, dim3, ch)
    # unet begins

    nfeatures = [2**feat*32 for feat in np.arange(6)]
    depth = len(nfeatures)    

    conv_ptr = []

    # input layer
    inputs = Input(input_size)

    # step down convolutional layers 
    pool = inputs
    for depth_cnt in xrange(depth):

        conv = Conv2D(nfeatures[depth_cnt], (3,3), 
                      padding='same', 
                      activation='relu',
                      kernel_initializer='he_normal')(pool)
        conv = Conv2D(nfeatures[depth_cnt], (3,3), 
                      padding='same', 
                      activation='relu',
                      kernel_initializer='he_normal')(conv)

        conv = BN(axis=-1, momentum=0.95, epsilon=0.001)(conv)
        conv = Dropout(rate=0.0)(conv)

        conv_ptr.append(conv)

        # Only maxpool till penultimate depth
        if depth_cnt < depth-1:

            # If size of input is odd, only do a 3x3 max pool
            xres = conv.shape.as_list()[1]
            if (xres % 2 == 0):
                pooling_size = (2,2)
            elif (xres % 2 == 1):
                pooling_size = (3,3)

            pool = MaxPooling2D(pool_size=pooling_size)(conv)


    # step up convolutional layers
    for depth_cnt in xrange(depth-2,-1,-1):

        deconv_shape = conv_ptr[depth_cnt].shape.as_list()
        deconv_shape[0] = None

        # If size of input is odd, then do a 3x3 deconv  
        if (deconv_shape[1] % 2 == 0):
            unpooling_size = (2,2)
        elif (deconv_shape[1] % 2 == 1):
            unpooling_size = (3,3)

        up = concatenate([Conv2DTranspose(nfeatures[depth_cnt],(3,3),
                          padding='same',
                          strides=unpooling_size,
                          output_shape=deconv_shape)(conv),
                          conv_ptr[depth_cnt]], 
                          axis=3)

        conv = Conv2D(nfeatures[depth_cnt], (3,3), 
                      padding='same', 
                      activation='relu',
                      kernel_initializer='he_normal')(up)
        conv = Conv2D(nfeatures[depth_cnt], (3,3), 
                      padding='same', 
                      activation='relu',
                      kernel_initializer='he_normal')(conv)

        conv = BN(axis=-1, momentum=0.95, epsilon=0.001)(conv)
        conv = Dropout(rate=0.00)(conv)

    # combine features
    recon = Conv2D(1, (1,1), padding='same', activation='sigmoid')(conv)

    model = Model(inputs=[inputs], outputs=[recon])
    plot_model(model, to_file='unet2d.png',show_shapes=True)
    
    return model


if __name__ == '__main__':
  
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    img_size = (288,288,1)
    model = unet_2d_model(img_size)
    print(model.summary())


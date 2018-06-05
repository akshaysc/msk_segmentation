# Authors:
# Akshay Chaudhari and Zhongnan Fang
# May 2018
# akshaysc@stanford.edu

import math
import keras.callbacks as kc  
import numpy as np
from sklearn.metrics import confusion_matrix

# Suppress divide by zero warning
np.seterr(divide='ignore', invalid='ignore')


# learning rate schedule
# Implementing a step decay for now
def step_decay(epoch):
    initial_lrate = 1e-4
    drop = 0.8
    epochs_drop = 1.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


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


def calc_cv(y_true, y_pred, thresh=0.01):

    recon = np.squeeze(y_true)
    pred = np.squeeze(y_pred)
    y_pred = (y_pred > thresh)*y_pred

    cv = 100*np.std([np.sum(y_true), np.sum(y_pred)])/np.mean([np.sum(y_true), np.sum(y_pred)])
    return cv

def calc_vd(y_true, y_pred, thresh=0.01):

    recon = np.squeeze(y_true)
    pred = np.squeeze(y_pred)
    y_pred = (y_pred > thresh)*y_pred

    vd = 100*(np.sum(y_pred) - np.sum(y_true))/np.sum(y_true)
    return vd

def calc_voe(y_true, y_pred, thresh=0.01):

  recon = np.squeeze(y_true)
  pred = np.squeeze(y_pred)
  y_pred = (y_pred > thresh)*y_pred
  
  # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
  TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
  # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
  TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
  # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
  FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
  # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
  FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))
  
  cm = np.array([[TP,FN], [FP,TP]])

  union = TP+FP+FN
  inter = TP

  voe = 100*(1-inter/union)
  # print('%0.2d, %0.2d, %0.2d, %0.2d, %0.2d, %0.2d, %0.2d' % (TP, FN, FP, TN, union, inter, voe))

  return voe

# During test time since it includes a discontinuity
def calc_dice(y_true, y_pred, thresh=0.01):
    
    recon = np.squeeze(y_true)
    pred = np.squeeze(y_pred)
    y_pred = (y_pred > thresh)*y_pred

    szp = y_pred.shape
    img_len = szp[1]*szp[2]*szp[3]

    y_true = np.reshape(y_true,(-1,img_len))
    y_pred = np.reshape(y_pred,(-1,img_len))

    ovlp = np.sum(y_true*y_pred,axis=-1)

    mu = 1e-07
    dice = (2.0 * ovlp + mu) / (np.sum(y_true,axis=-1) + np.sum(y_pred,axis=-1) + mu)

    return dice
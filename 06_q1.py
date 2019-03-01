from __future__ import absolute_import, division, print_function

import argparse
import os
import shutil
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib import eager as tfe
from tensorflow.keras import layers
import time

import util
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

IMAGENET_MEAN = np.array([123.68, 116.81, 103.94],dtype="float32")


class CaffeNet(keras.Model):
    def __init__(self, num_classes=10):
        super(CaffeNet, self).__init__(name='CaffeNet')
        self.num_classes = num_classes
        self.conv1 = layers.Conv2D(filters=96, #Need to add 4 
                                   strides=[4,4],
                                   kernel_size=[11, 11],
                                   padding="valid",
                                   activation='relu')
        self.pool1 = layers.MaxPool2D(pool_size=(3, 3), strides=(2,2))
        self.conv2 = layers.Conv2D(filters=256,
                                   kernel_size=[5, 5],
                                   padding="same",
                                   activation='relu')
        self.pool2 = layers.MaxPool2D(pool_size=(3, 3), strides=(2,2))
        self.conv3 = layers.Conv2D(filters=384,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.conv4 = layers.Conv2D(filters=384,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.conv5 = layers.Conv2D(filters=256,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.pool3 = layers.MaxPool2D(pool_size=(3, 3), strides=(2,2))
        self.flat = layers.Flatten()
        self.dense1 = layers.Dense(4096, activation='relu')
        self.dropout1 = layers.Dropout(rate=0.5)
        self.dense2 = layers.Dense(4096, activation='relu')
        self.dropout2 = layers.Dropout(rate=0.5)
        self.dense3 = layers.Dense(num_classes)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        flat_x = self.flat(x)
        out = self.dense1(flat_x)
        out = self.dropout1(out, training=training)
        out = self.dense2(out)
        out = self.dropout2(out, training=training)
        out = self.dense3(out)
        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape = [shape[0], self.num_classes]
        return tf.TensorShape(shape)

def plotWeights(weights, num):
    plt.figure(figsize = (12,20))
    gs1 = gridspec.GridSpec(8,12)
    gs1.update(wspace=0.025, hspace=0.025) # set the spacing between axes. 

    for i in range(96):
    # i = i + 1 # grid spec indexes from 0
        ax1 = plt.subplot(gs1[i])
        plt.axis('off')
        ax1.set_aspect('equal')
        resp = weights[:,:,:,i]
        resp_min = resp.min(axis=(0,1),keepdims=True)
        resp_max = resp.max(axis=(0,1),keepdims=True)
        resp = (resp-resp_min)/(resp_max-resp_min)
        ax1.imshow(resp)
    # plt.title("Visualization of conv1 of CaffeNet")
    plt.show()
    # plt.savefig("data/kernel-"+str(i)+".png")

def main():
    model = CaffeNet(num_classes=len(CLASS_NAMES))
    model.build(input_shape=(1,224,224,3))
    filter1=[]
    filter2=[]
    filter3=[]
    for i in range(0,31):
        model.load_weights("checkpoints/"+str(i)+"-03-weights.h5")
        conv1_W = model.conv1.get_weights()[0]
        plotWeights(conv1_W,i)
        
if __name__ == '__main__':
    tf.enable_eager_execution()
    main()
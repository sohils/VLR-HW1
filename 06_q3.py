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
import random

import util
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE

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
    
    def do_pool5(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        flat_x = self.flat(x)
        return flat_x
    
    def do_fc7(self, inputs, training=False):
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
        return out

def main():
    parser = argparse.ArgumentParser(description='TensorFlow Pascal Example')
    parser.add_argument('--data-dir', type=str, default='./data/VOCdevkit/VOC2007/',
                    help='Path to PASCAL data storage')
    args = parser.parse_args()

    caffe_model = CaffeNet(num_classes=len(CLASS_NAMES))
    caffe_model.build(input_shape=(1,224,224,3))
    caffe_model.load_weights("checkpoints/30-03-weights.h5")


    test_images, test_labels, test_weights = util.load_pascal(args.data_dir,
                                                              class_names=CLASS_NAMES,
                                                              split='test')

    max_index = test_images.shape[0]
    indices = random.sample(range(0, max_index), 1000)
    test_images = test_images[indices]
    test_labels = test_labels[indices]
    test_weights = test_weights[indices]       

    test_images = test_images - IMAGENET_MEAN
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels, test_weights))
    test_dataset = test_dataset.map(lambda x,y,z: util.center_crop(x,y,z)).batch(1)
    
    feature_vector_caffe_fc7=[]

    for index, (images, labels, weights) in enumerate(test_dataset):
        feature_vector_caffe_fc7.append(caffe_model.do_fc7(images))
    
    feature_vector_caffe_fc7=np.asarray(feature_vector_caffe_fc7)

    caffe_fc7_tsne_projection = TSNE(n_components=2).fit_transform(feature_vector_caffe_fc7)
    np.savez("tsne.npz", projections=caffe_fc7_tsne_projection, labels=test_labels)
    # plt.scatter(caffe_fc7_tsne_projection[:,0], caffe_fc7_tsne_projection[:,1])

if __name__ == '__main__':
    tf.enable_eager_execution()
    main()
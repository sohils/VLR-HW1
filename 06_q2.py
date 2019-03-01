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
from PIL import Image

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

class VGG16(keras.Model):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__(name='VGG16')
        self.num_classes = num_classes
        self.conv1_1 = layers.Conv2D(filters=64,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.conv1_2 = layers.Conv2D(filters=64,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.pool1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2,2))
        self.conv2_1 = layers.Conv2D(filters=128,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.conv2_2 = layers.Conv2D(filters=128,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.pool2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2,2))
        self.conv3_1 = layers.Conv2D(filters=256,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.conv3_2 = layers.Conv2D(filters=256,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.conv3_3 = layers.Conv2D(filters=256,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.pool3 = layers.MaxPool2D(pool_size=(2, 2), strides=(2,2))
        self.conv4_1 = layers.Conv2D(filters=512,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.conv4_2 = layers.Conv2D(filters=512,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.conv4_3 = layers.Conv2D(filters=512,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.pool4 = layers.MaxPool2D(pool_size=(2, 2), strides=(2,2))
        self.conv5_1 = layers.Conv2D(filters=512,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.conv5_2 = layers.Conv2D(filters=512,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.conv5_3 = layers.Conv2D(filters=512,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.pool5 = layers.MaxPool2D(pool_size=(2, 2), strides=(2,2))
        self.flat = layers.Flatten()
        self.dense1 = layers.Dense(4096, activation='relu')
        self.dropout1 = layers.Dropout(rate=0.5)
        self.dense2 = layers.Dense(4096, activation='relu')
        self.dropout2 = layers.Dropout(rate=0.5)
        self.dense3 = layers.Dense(num_classes)

    def call(self, inputs, training=False):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)
        flat_x = self.flat(x)
        out = self.dense1(flat_x)
        out = self.dropout1(out, training=training)
        out = self.dense2(out)
        out = self.dropout2(out, training=training)
        out = self.dense3(out)
        return out
    
    def do_pool5(self, inputs, training=False):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)
        flat_x = self.flat(x)
        return flat_x

    def do_fc7(self, inputs, training=False):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)
        flat_x = self.flat(x)
        out = self.dense1(flat_x)
        out = self.dropout1(out, training=training)
        out = self.dense2(out)
        out = self.dropout2(out, training=training)
        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape = [shape[0], self.num_classes]
        return tf.TensorShape(shape)

def main():
    parser = argparse.ArgumentParser(description='TensorFlow Pascal Example')
    parser.add_argument('--data-dir', type=str, default='./data/VOCdevkit/VOC2007/',
                    help='Path to PASCAL data storage')
    args = parser.parse_args()

    vgg_model = VGG16(num_classes=len(CLASS_NAMES))
    caffe_model = CaffeNet(num_classes=len(CLASS_NAMES))

    vgg_model.build(input_shape=(1,224,224,3))
    caffe_model.build(input_shape=(1,224,224,3))

    vgg_model.load_weights("checkpoints/2510-05-weights.h5")
    caffe_model.load_weights("checkpoints/30-03-weights.h5")

    ground_image_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'motorbike',
               'pottedplant', 'sheep', 'sofa', 'tvmonitor']
    ground_images = []
    for index, line in enumerate(ground_image_names):
        img = np.array(Image.open("q6_2_images/"+line.strip()+".jpg").resize((256,256)))
        ground_images.append(img)
    ground_images = np.asarray(ground_images)

    ground_images = ground_images - IMAGENET_MEAN
    ground_dataset = tf.data.Dataset.from_tensor_slices((ground_images))
    ground_dataset = ground_dataset.map(lambda x: util.center_crop_2(x)).batch(1)
    ground_feature_vector_caffe_pool5=[]
    ground_feature_vector_caffe_fc7=[]
    ground_feature_vector_vgg_pool5=[]
    ground_feature_vector_vgg_fc7=[]
    for index, image in enumerate(ground_dataset):
        ground_feature_vector_caffe_pool5.append(caffe_model.do_pool5(image).numpy().flatten())
        ground_feature_vector_caffe_fc7.append(caffe_model.do_fc7(image).numpy().flatten())
        ground_feature_vector_vgg_pool5.append(vgg_model.do_pool5(image).numpy().flatten())
        ground_feature_vector_vgg_fc7.append(vgg_model.do_fc7(image).numpy().flatten())

    ground_feature_vector_caffe_pool5 = np.asarray(ground_feature_vector_caffe_pool5)
    ground_feature_vector_caffe_fc7 = np.asarray(ground_feature_vector_caffe_fc7)
    ground_feature_vector_vgg_pool5 = np.asarray(ground_feature_vector_vgg_pool5)
    ground_feature_vector_vgg_fc7 = np.asarray(ground_feature_vector_vgg_fc7)

    print("Extracted ground images and vectors")

    test_images, test_labels, test_weights = util.load_pascal(args.data_dir,
                                                              class_names=CLASS_NAMES,
                                                              split='test')
    test_images = test_images - IMAGENET_MEAN
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels, test_weights))
    test_dataset = test_dataset.map(lambda x,y,z: util.center_crop(x,y,z)).batch(1)

    test_feature_vector_caffe_pool5=[]
    test_feature_vector_caffe_fc7=[]
    test_feature_vector_vgg_pool5=[]
    test_feature_vector_vgg_fc7=[]

    print("Extracted ground images and vectors")

    for index, (images, labels, weights) in enumerate(train_dataset):
        feature_vector_caffe_pool5 = caffe_model.do_pool5(images).flatten()
        feature_vector_caffe_fc7 = caffe_model.do_fc7(images).flatten()
        feature_vector_vgg_pool5 = vgg_model.do_pool5(images).flatten()
        feature_vector_vgg_fc7 = vgg_model.do_fc7(images).flatten()

        test_feature_vector_caffe_pool5.append(np.sum(np.square(ground_feature_vector_caffe_pool5 - feature_vector_caffe_pool5), axis=1))
        test_feature_vector_caffe_fc7.append(np.sum(np.square(ground_feature_vector_caffe_fc7 - feature_vector_caffe_fc7), axis=1))
        test_feature_vector_vgg_pool5.append(np.sum(np.square(ground_feature_vector_vgg_pool5 - feature_vector_vgg_pool5), axis=1))
        test_feature_vector_vgg_fc7.append(np.sum(np.square(ground_feature_vector_vgg_fc7 - feature_vector_vgg_fc7), axis=1))

        if(index == 0):
            print(test_feature_vector_caffe_pool5.shape)

    print("After distance calculations, size:", test_feature_vector_caffe_pool5.shape)

    test_feature_vector_caffe_pool5 = np.asarray(test_feature_vector_caffe_pool5).argsort(axis=0)[0:4,:]
    test_feature_vector_caffe_fc7 = np.asarray(test_feature_vector_caffe_fc7).argsort(axis=0)[0:4,:]
    test_feature_vector_vgg_pool5 = np.asarray(test_feature_vector_vgg_pool5).argsort(axis=0)[0:4,:]
    test_feature_vector_vgg_fc7 = np.asarray(test_feature_vector_vgg_fc7).argsort(axis=0)[0:4,:]
    
    print("Found min distance, size:", test_feature_vector_caffe_pool5.shape)

    for i in range(test_feature_vector_caffe_pool5.shape[0]):
        for j in range(test_feature_vector_caffe_pool5.shape[1]):
            Image.fromarray(test_images[test_feature_vector_caffe_pool5[i,j]], mode='RGB').save("results/Caffe_Pool5_"+str(j)+"_"+str(i)+"_nearest.png")
            Image.fromarray(test_images[test_feature_vector_caffe_fc7[i,j]], mode='RGB').save("results/Caffe_FC7_"+str(j)+"_"+str(i)+"_nearest.png")
            Image.fromarray(test_images[test_feature_vector_vgg_pool5[i,j]], mode='RGB').save("results/VGG_Pool5_"+str(j)+"_"+str(i)+"_nearest.png")
            Image.fromarray(test_images[test_feature_vector_vgg_fc7[i,j]], mode='RGB').save("results/VGG_FC7_"+str(j)+"_"+str(i)+"_nearest.png")


if __name__ == '__main__':
    tf.enable_eager_execution()
    main()

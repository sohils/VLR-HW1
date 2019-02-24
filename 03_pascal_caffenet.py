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
        self.pool1 = layers.MaxPool2D(pool_size=(3, 2))
        self.conv2 = layers.Conv2D(filters=256,
                                   kernel_size=[5, 5],
                                   padding="same",
                                   activation='relu')
        self.pool2 = layers.MaxPool2D(pool_size=(3, 2))
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
        self.pool3 = layers.MaxPool2D(pool_size=(3, 2))
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

def test(model, dataset):
    preds = []
    gts = []
    valids = []
    test_loss = tfe.metrics.Mean()
    test_accuracy = tfe.metrics.BinaryAccuracy(threshold=0.7)
    for batch, (images, labels, weights) in enumerate(dataset):
        logits = model(images)
        loss_value = tf.losses.sigmoid_cross_entropy(labels, logits)
        prediction = tf.nn.sigmoid(logits)
        test_loss(loss_value,weights=weights)
        test_accuracy(labels=tf.cast(labels,tf.bool), predictions=prediction, weights=weights)
        preds.append(prediction.numpy())
        gts.append(labels.numpy())
        valids.append(weights.numpy())
    # Stack all the predicitons, labels and weights
    preds = np.vstack(preds)
    gts = np.vstack(gts)
    valids = np.vstack(valids)
    # Calculate mAP
    AP, mAP = util.eval_dataset_map_helper(gts, preds, valids)
    return test_loss.result(), test_accuracy.result(), mAP

def logging_variable(name, value):
    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar(name, value)

def main():
    parser = argparse.ArgumentParser(description='TensorFlow Pascal Example')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--decay', type=float, default=0.001,
                        help='decay')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before'
                             ' logging training status')
    parser.add_argument('--eval-interval', type=int, default=20,
                        help='how many batches to wait before'
                             ' evaluate the model')
    parser.add_argument('--log-dir', type=str, default='tb',
                        help='path for logging directory')
    parser.add_argument('--data-dir', type=str, default='./data/VOCdevkit/VOC2007/',
                        help='Path to PASCAL data storage')
    parser.add_argument('--chkpt-dir', type=str, default='./checkpoint/',
                        help='Path to model checkpoint storage')
    args = parser.parse_args()
    util.set_random_seed(args.seed)
    sess = util.set_session()

    train_images, train_labels, train_weights = util.load_pascal(args.data_dir,
                                                                 class_names=CLASS_NAMES,
                                                                 split='trainval')
    test_images, test_labels, test_weights = util.load_pascal(args.data_dir,
                                                              class_names=CLASS_NAMES,
                                                              split='val')
    train_images = train_images - IMAGENET_MEAN
    test_images = test_images - IMAGENET_MEAN

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels, train_weights))
    train_dataset = util.data_augmentation(train_dataset,args.seed)
    # print(train_dataset.output_shape)
    train_dataset = train_dataset.shuffle(10000).batch(args.batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels, test_weights))
    test_dataset = test_dataset.map(lambda x,y,z: util.center_crop(x,y,z))
    # print(test_dataset.output_shape)
    # test_dataset = util.data_augmentation(test_dataset,args.seed)
    test_dataset = test_dataset.shuffle(10000).batch(args.batch_size)

    del(train_images)
    del(test_images)

    model = CaffeNet(num_classes=len(CLASS_NAMES))

    # Logging block
    logdir = os.path.join(args.log_dir,
                          datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)
    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()

    start_time = time.time()

    ## TODO write the training and testing code for multi-label classification
    global_step = tf.train.get_or_create_global_step()
   
    # Defining a decaying learning rate.
    learning_rate = tf.train.exponential_decay(learning_rate=args.lr, global_step=global_step, 
                                        decay_steps=5000, decay_rate=0.5)
    # SGD + Momentum optimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=args.momentum)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    train_log = {'iter': [], 'loss': [], 'accuracy': []}
    test_log = {'iter': [], 'loss': [], 'map': [], 'accuracy': []}
    for ep in range(args.epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.BinaryAccuracy(threshold=0.7)
        for batch, (images, labels, weights) in enumerate(train_dataset):
            loss_value, grads = util.cal_grad(model,
                                              loss_func=tf.losses.sigmoid_cross_entropy,
                                              inputs=images,
                                              targets=labels,
                                              weights=weights)
            optimizer.apply_gradients(zip(grads,
                                          model.trainable_variables),
                                      global_step)
            epoch_loss_avg(loss_value, weights=weights)
            pred = tf.nn.sigmoid(model(images))
            epoch_accuracy(predictions=pred,labels=tf.cast(labels,tf.bool),weights=weights)
            if global_step.numpy() % args.log_interval == 0:
                print('Epoch: {0:d}/{1:d} Iteration:{2:d}  Training Loss:{3:.4f}  '
                      'Training Accuracy:{4:.4f}'.format(ep,
                                                         args.epochs,
                                                         global_step.numpy(),
                                                         epoch_loss_avg.result(),
                                                         epoch_accuracy.result()))
                train_log['iter'].append(global_step.numpy())
                train_log['loss'].append(epoch_loss_avg.result())
                train_log['accuracy'].append(epoch_accuracy.result())
                # Logging for TensorFlow
                logging_variable('train_loss',epoch_loss_avg.result())
                logging_variable('train_accuracy',epoch_accuracy.result())
            if global_step.numpy() % args.eval_interval == 0:
                test_loss, test_accuracy, mAP = test(model, test_dataset)
                test_log['iter'].append(global_step.numpy())
                test_log['loss'].append(test_loss)
                test_log['accuracy'].append(test_accuracy)
                test_log['map'].append(mAP)
                # Logging for TensorFlow
                logging_variable('test_mAP',mAP)
                logging_variable('test_loss',test_loss)
                logging_variable('test_accuracy',test_accuracy)
        if ep%2 ==0 :
            checkpoint.save("./checkpoints/ckpt")

    model.summary()
    end_time = time.time()
    print('Elapsed time: {0:.3f}s'.format(end_time - start_time))
    
    np.save("03_training.npy", train_log)
    np.save("03_test.npy", test_log)
    checkpoint.save("./checkpoints/ckpt")

    AP, mAP = util.eval_dataset_map(model, test_dataset)
    rand_AP = util.compute_ap(
        test_labels, np.random.random(test_labels.shape),
        test_weights, average=None)
    print('Random AP: {} mAP'.format(np.mean(rand_AP)))
    gt_AP = util.compute_ap(test_labels, test_labels, test_weights, average=None)
    
    print('GT AP: {} mAP'.format(np.mean(gt_AP)))
    print('Obtained {} mAP'.format(mAP))
    print('Per class:')
    for cid, cname in enumerate(CLASS_NAMES):
        print('{}: {}'.format(cname, util.get_el(AP, cid)))


if __name__ == '__main__':
    tf.enable_eager_execution()
    main()

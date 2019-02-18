import numpy as np
import sklearn.metrics
from tensorflow import keras
from PIL import Image
import xml.etree.ElementTree as xmletree
import tensorflow as tf

def set_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    keras.backend.set_session(session)
    return session


def set_random_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)


def load_pascal(data_dir, class_names, split='train'):
    """
    Function to read images from PASCAL data folder.
    Args:
        data_dir (str): Path to the VOC2007 directory.
        class_names (list): list of class names
        split (str): train/val/trainval split to use.
    Returns:
        images (np.ndarray): Return a np.float32 array of
            shape (N, H, W, 3), where H, W are 256px each,
            and each image is in RGB format.
        labels (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are active in that image.
        weights: (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are confidently labeled and 0s for classes that
            are ambiguous.
    """
    ## TODO Implement this function
    images = []
    labels = []
    weights = []

    with open(data_dir+"ImageSets/Main/"+split+".txt") as fp:
        lines = [line.strip() for line in fp.readlines()]

    for line in lines:
        label = np.zeros((len(class_names)))
        weight = np.ones((len(class_names)))
        # img = keras.preprocessing.image.load_img(data_dir+"ImageSets/"+line.strip()+".jpg")
        img = np.array(Image.open(data_dir+"JPEGImages/"+line.strip()+".jpg").resize((256,256)))
        images.append(img)
        e = xmletree.parse(data_dir+"Annotations/"+line.strip()+".xml").getroot()
        for obj in e.findall('object'):
            tag = class_names.index(obj.find('name').text)
            label[tag]=1
        labels.append(label)
        weights.append(weight)

    images = np.asarray(images)
    labels = np.asarray(labels)
    weights = np.asarray(weights)
    return images, labels, weights


def cal_grad(model, loss_func, inputs, targets, weights=1.0):
    """
    Return the loss value and gradients
    Args:
         model (keras.Model): model
         loss_func: loss function to use
         inputs: image inputs
         targets: labels
         weights: weights of the samples
    Returns:
         loss and gradients
    """

    with tf.GradientTape() as tape:
        logits = model(inputs)
        loss_value = loss_func(targets, logits, weights)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def compute_ap(gt, pred, valid, average=None):
    """
    Compute the multi-label classification accuracy.
    Args:
        gt (np.ndarray): Shape Nx20, 0 or 1, 1 if the object i is present in that
            image.
        pred (np.ndarray): Shape Nx20, probability of that object in the image
            (output probablitiy).
        valid (np.ndarray): Shape Nx20, 0 if you want to ignore that class for that
            image. Some objects are labeled as ambiguous.
    Returns:
        AP (list): average precision for all classes
    """
    nclasses = gt.shape[1]
    AP = []
    for cid in range(nclasses):
        gt_cls = gt[:, cid][valid[:, cid] > 0].astype('float32')
        pred_cls = pred[:, cid][valid[:, cid] > 0].astype('float32')
        # As per PhilK. code:
        # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
        pred_cls -= 1e-5 * gt_cls
        ap = sklearn.metrics.average_precision_score(
            gt_cls, pred_cls, average=average)
        AP.append(ap)
    return AP


def eval_dataset_map(model, dataset):
    """
    Evaluate the model with the given dataset
    Args:
         model (keras.Model): model to be evaluated
         dataset (tf.data.Dataset): evaluation dataset
    Returns:
         AP (list): Average Precision for all classes
         MAP (float): mean average precision
    """
    ## TODO implement the code here
    return AP, mAP


def get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arr

def data_augmentation(images, labels, weights, seed):
    images = tf.image.random_flip_left_right(images,seed=seed)
    images = tf.image.random_flip_up_down(images,seed=seed)
    images = tf.image.random_crop(images, images.shape)
    images = tf.image.random_brightness(images, max_delta=0.75)
    return (images, labels, weights)
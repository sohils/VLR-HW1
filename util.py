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

    filename = data_dir+"ImageSets/Main/"+split+".txt"
    with open(filename) as fp:
            lines = [line.strip() for line in fp.readlines()]
    for index, line in enumerate(lines):
        # if(index == 20): # Please delete this
        #     break
        img = np.array(Image.open(data_dir+"JPEGImages/"+line.strip()+".jpg").resize((256,256)))
        images.append(img)

    images = np.asarray(images)
    labels = np.zeros((images.shape[0],len(class_names)))
    weights = np.zeros((images.shape[0],len(class_names)))

    for index, class_name in enumerate(class_names):
        filename = data_dir+"ImageSets/Main/"+class_name+"_"+split+".txt"
        with open(filename) as fp:
            lines = [line.strip() for line in fp.readlines()]
        for image_index,line in enumerate(lines):
            # if(image_index == 20): # Please delete this
            #     break
            words = line.split()
            if( words[1] == '1'):
                labels[image_index,index] = 1
                weights[image_index,index] = 1
            elif( words[1] == '0'):
                labels[image_index,index] = 1
                weights[image_index,index] = 0
            elif( words[1] == '-1'):
                labels[image_index,index] = 0
                weights[image_index,index] = 1
    return images.astype("float32"), labels.astype("int32"), weights.astype("int32")


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
        logits = model(inputs, training=True)
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
    preds = []
    gts = []
    valids = []
    for batch, (images, labels, weights) in enumerate(dataset):
        preds.append(tf.nn.sigmoid(model(images)).numpy())
        gts.append(labels)
        valids.append(weights)
    preds = np.vstack(preds)
    gts = np.vstack(gts)
    valids = np.vstack(valids)
    return (eval_dataset_map_helper(gt=gts,pred=preds, valid = valids))


def eval_dataset_map_helper(gt, pred, valid):
    AP = compute_ap(gt, pred, valid)
    mAP = np.average(AP)
    return AP, mAP

def get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arr

def data_augmentation(dataset, seed):
    dataset = dataset.concatenate(dataset.map(lambda x,y,z: data_augmentation_flip_left_right(x,y,z,seed)))
    # dataset = dataset.concatenate(dataset.map(lambda x,y,z: data_augmentation_flip_up_down(x,y,z,seed)))
    dataset = dataset.map(lambda x,y,z: data_augmentation_crop(x,y,z,seed))
    return dataset

def data_augmentation_flip_left_right(images, labels, weights, seed):
    images = tf.image.random_flip_left_right(images,seed=seed)
    return (images, labels, weights)

def data_augmentation_flip_up_down(images, labels, weights, seed):
    images = tf.image.random_flip_up_down(images,seed=seed)
    return (images, labels, weights)

def data_augmentation_crop(images, labels, weights, seed):
    images = tf.image.random_crop(images, size = [224,224,3], seed = seed)
    # images = tf.image.resize_images(images, size = [256,256])
    return (images, labels, weights)

def center_crop(images, labels, weights):
    images = tf.image.central_crop(images, (224/256))
    return (images, labels, weights)
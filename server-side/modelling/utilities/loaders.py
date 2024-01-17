import csv
import numpy as np
import tqdm
import pickle
import json
import os
import pandas as pd

from pathlib import Path
from splitfolders import ratio

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.text import tokenizer_from_json


def device_exists():
    """
    returns true if gpu device exists
    """

    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        return False
    return True

def create_image_set(root_dir: str, img_dims: tuple=(256, 256)):
    temp = sorted(os.listdir(root_dir))

    # creates new copies of the subdirectories of train, cross, and
    # testing folders under each class/label subdirectory e.g. 
    # Amoeba will have now train, cross, and testing folders in it
    sub_dirs = Path(root_dir)
    output_dir = f'{root_dir[:-1]}_Split'
    ratio(sub_dirs, output=output_dir, seed=0, ratio=(0.7, 0.15, 0.15), group_prefix=None)

    # augments the unbalanced image data we currently have
    # by rotating, flipping, distorting images to produce
    # more of a balanced image set
    gen = ImageDataGenerator(
        # instead of our rgb values being between 0 and 255 doing 
        # this rescales the rgb values between 0 and 1
        rescale=1.0 / 255,

        # degree range for random rotations.
        rotation_range=10,

        # Randomly flip inputs horizontally
        horizontal_flip=True,

        vertical_flip=True,

        # values lie between 0 being dark and 1 being bright
        brightness_range=[0.3, 0.8]
    )

    # gen.flow_from_directory actually returns a generator object
    # which recall we can use the next() with to get the next element
    train_gen = gen.flow_from_directory(
        # this arg should contain one subdirectory per class. Any PNG, JPG, 
        # BMP, PPM or TIF images inside each of the subdirectories directory 
        # tree will be included in the generator
        f'{output_dir}/train',
        target_size=img_dims,

        # means the labels/targets produced by generator will be
        # a one encoding of each of our different classes e.g.
        # amoeba will have [1, 0, 0, 0, 0, 0, 0, 0]
        class_mode='categorical',
        subset='training',
        batch_size=128
    )

    cross_gen = gen.flow_from_directory(
        f'{output_dir}/val',
        target_size=img_dims,
        class_mode='categorical',
        shuffle=False
    )

    test_gen = gen.flow_from_directory(
        f'{output_dir}/test',
        target_size=img_dims,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, cross_gen, test_gen

def create_metrics_df(train_metric_values, val_metric_values, test_metric_values):
    """
    creates a metrics dataframe
    """

    train_acc, train_prec, train_rec, train_f1 = train_metric_values
    val_acc, val_prec, val_rec, val_f1 = val_metric_values
    test_acc, test_prec, test_rec, test_f1 = test_metric_values

    metrics_df = pd.DataFrame({
        'data_split': ['training', 'validation', 'testing'],
        'accuracy': [train_acc, val_acc, test_acc], 
        'precision': [train_prec, val_prec, test_prec], 
        'recall': [train_rec, val_rec, test_rec], 
        'f1-score': [train_f1, val_f1, test_f1]
    })

    return metrics_df

def create_classified_df(train_conf_matrix, val_conf_matrix, test_conf_matrix, train_labels, val_labels, test_labels):
    """
    creates a dataframe that represents all classified and 
    misclassified values
    """

    num_right_cm_train = train_conf_matrix.trace()
    num_right_cm_val = val_conf_matrix.trace()
    num_right_cm_test = test_conf_matrix.trace()

    num_wrong_cm_train = train_labels.shape[0] - num_right_cm_train
    num_wrong_cm_val = val_labels.shape[0] - num_right_cm_val
    num_wrong_cm_test = test_labels.shape[0] - num_right_cm_test

    classified_df = pd.DataFrame({
        'data_split': ['training', 'validation', 'testing'],
        'classified': [num_right_cm_train, num_right_cm_val, num_right_cm_test], 
        'misclassified': [num_wrong_cm_train, num_wrong_cm_val, num_wrong_cm_test]}, 
        index=["training set", "validation set", "testing set"])
    
    return classified_df

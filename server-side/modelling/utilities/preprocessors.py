from PIL import Image

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, LabelEncoder

import pandas as pd
import numpy as np
import tensorflow as tf



def one_hot_encode(sparse_labels):
    """
    one hot encodes a sparse multi-class label
    e.g. if in sparse labels contain [0 1 2 0 3]
    one hot encoding would be
    [[1 0 0 0]
    [0 1 0 0]
    [0 0 1 0]
    [1 0 0 0]
    [0 0 0 1]]

    used only during training and validation
    """

    n_unique = np.unique(sparse_labels).shape[0]
    one_hot_encoding = tf.one_hot(sparse_labels, depth=n_unique)

    return one_hot_encoding

def decode_one_hot(Y_preds):
    """
    whether for image, sentiment, or general classification
    this function takes in an (m x 1) or (m x n_y) matrix of
    the predicted values of a classifier

    e.g. if binary the input Y_preds would be 
    [[0 1]
    [1 0]
    [1 0]
    [0 1]
    [1 0]
    [1 0]]

    if multi-class the Y_preds for instance would be...

    [[0 0 0 1]
    [1 0 0 0
    [0 0 1 0]
    ...
    [0 1 0 0]]

    what this function does is it takes the argmax along the
    1st dimension/axis, and once decoded would be just two
    binary categorial values e.g. 0 or 1 or if multi-class
    0, 1, 2, or 3

    main args:
        Y_preds - 

    used during training, validation, and testing/deployment
    """

    # check if Y_preds is multi-class by checking if shape
    # of matrix is (m, n_y), (m, m, n_y), or just m
    if len(Y_preds.shape) >= 2:
        # take the argmax if Y_preds are multi labeled
        sparse_categories = np.argmax(Y_preds, axis=1)

    return sparse_categories

def re_encode_sparse_labels(sparse_labels, new_labels: list=['DER', 'APR', 'NDG']):
    """
    sometimes a dataset will only have its target values 
    be sparse values such as 0, 1, 2 right at the start
    so this function re encodes these sparse values/labels
    to a more understandable representation

    upon reencoding this can be used by other encoders
    such as encode_features() which allows us to save
    the encoder to be used later on in model training

    used only during training and validation
    """

    # return use as index the sparse_labels to the new labels
    v_func = np.vectorize(lambda sparse_label: new_labels[sparse_label])
    re_encoded_labels = v_func(sparse_labels)

    return re_encoded_labels

def translate_labels(labels, translations: dict={'DER': 'Derogatory', 
                                                 'NDG': 'Non-Derogatory', 
                                                 'HOM': 'Homonym', 
                                                 'APR': 'Appropriative'}):
    """
    transforms an array of shortened versions of the
    labels e.g. array(['DER', 'NDG', 'DER', 'HOM', 'APR', 
    'DER', 'NDG', 'HOM', 'HOM', 'HOM', 'DER', 'DER', 'NDG', 
    'DER', 'HOM', 'DER', 'APR', 'APR', 'DER'] to a more lengthened
    and understandable version to potentially send back to client
    e.g. array(['DEROGATORY', NON-DEROGATORY, 'DEROGATORY', 'HOMONYM',
    'APPROPRIATIVE', ...])

    used during training, validation, and testing/deployment
    """

    v_func = np.vectorize(lambda label: translations[label])
    translated_labels = v_func(labels)
    return translated_labels

def encode_image(image_path: str, dimensions: tuple=(256, 256)):
    """
    encodes an image to a 3D matrix that can be used to
    feed as input to a convolutional model

    used primarily during testing/deployment to encode
    given image from client but image encoder for training
    is done by create_image_set() from loaders.py
    """

    # open image
    img = Image.open(image_path)

    # if there are given dimensions for new image size resize image
    if dimensions != None:
        img = img.resize(size=dimensions)

    # encode image by converting to numpy array
    encoded_img = np.asarray(img)

    # close image file
    img.close()

    return encoded_img


def standardize_image(encoded_img):
    """
    rescales an encoded image's values from 0 to 255 down
    to 0 and 1

    used primarily during testing/deployment to encode
    given image from client but image standardization for
    training is done by create_image_set() from loaders.py
    """

    rescaled_img = (encoded_img * 1.0) / 255

    return rescaled_img
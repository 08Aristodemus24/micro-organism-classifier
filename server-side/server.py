from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from datetime import datetime as dt

# ff. imports are for getting secret values from .env file
from pathlib import Path
import os

from modelling.utilities.preprocessors import (
    translate_labels,
    encode_image,
    standardize_image,
    re_encode_sparse_labels,
    decode_one_hot,
    activate_logits
)
from modelling.utilities.visualizers import (
    show_image
)

import tensorflow as tf
import numpy as np

# configure location of build file and the static html template file
app = Flask(__name__, template_folder='static')

# since simple html from url http://127.0.0.1:5000 requests to
# api endpoint at http://127.0.0.1:5000/ we must set the allowed
# origins or web apps with specific urls like http://127.0.0.1:5000
# to be included otherwise it will be blocked by CORS policy
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5000",])

models = []

def load_models():
    """
    prepares and loads sample input and custom model in
    order to use trained weights/parameters/coefficients
    """

    # recreate model architecture
    saved_model = tf.keras.models.load_model('./modelling/final/models/test_architecture-inception-v3_10_0.66.h5')
    models.append(saved_model)

load_models()



@app.route('/predict', methods=['POST'])
def predict():
    # extract raw data from client
    raw_files = request.files
    print(raw_files)

    image = raw_files.get('image')

    # preprocessing/encoding image stream into a matrix
    encoded_img = encode_image(image.stream)
    rescaled_img = standardize_image(encoded_img)
    
    # reshape the image since the model takes in an (m, 256, 256, 3)
    # input, or in this case a single (1, 256, 256, 3) input
    img_shape = rescaled_img.shape
    reshaped_img = np.reshape(rescaled_img, newshape=(1, img_shape[0], img_shape[1], img_shape[2]))
    
    # predictor
    logits = models[0].predict(reshaped_img)

    # decoding stage
    Y_preds = activate_logits(logits)
    Y_preds = decode_one_hot(Y_preds)
    final_preds = re_encode_sparse_labels(Y_preds, new_labels=['Amoeba', 'Euglena', 'Hydra', 'Paramecium', 'Rod_bacteria', 'Spherical_bacteria', 'Spiral_bacteria', 'Yeast'])
    print(final_preds)
    
    return jsonify({'prediction': final_preds.tolist()})
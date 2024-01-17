# model architecture will be defined here
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Activation, 
    Dropout, 
    Dense, 
    BatchNormalization,
    Conv2D, 
    MaxPooling2D, 
    Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.losses import CategoricalCrossentropy as cce_loss
from tensorflow.keras.metrics import CategoricalCrossentropy as cce_metric, CategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import keras_tuner as kt

import numpy as np


def load_baseline_a(n_classes):
    # define architecture
    model = Sequential([
        # build conv and poollayers
        Conv2D(filters=8,
            kernel_size=(5, 5),
            strides=(1, 1),
            kernel_regularizer=L2(0.8)),
        Activation(activation=tf.nn.relu),
        MaxPooling2D(pool_size=(2, 2),
            strides=(2, 2),
            padding='same'),
        Conv2D(filters=16,
            kernel_size=(5, 5),
            strides=(1, 1),
            kernel_regularizer=L2(0.8)),
        Activation(activation=tf.nn.relu),
        MaxPooling2D(pool_size=(2, 2),
            strides=(1, 1),
            padding='same'),

        # flatten pooled layers
        Flatten(),

        # build fully connected layers
        Dense(units=32),
        BatchNormalization(),
        Activation(activation=tf.nn.relu),
        Dense(units=n_classes),

    ], name='architecture-A')

    return model

def compile_model(raw_model, data, learning_rate):
    # define loss, optimizer, and metrics then compile
    opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
    loss = cce_loss(from_logits=True)
    metrics = [CategoricalAccuracy(), cce_metric(from_logits=True)]    
    raw_model.compile(optimizer=opt, loss=loss, metrics=metrics)
    raw_model(data)
    raw_model.summary()

    return raw_model

def train_model(compiled_model, training_data, validation_data, epochs, batch_size):
    # define checkpoint and early stopping callback to save
    # best weights at each epoch and to stop if there is no improvement
    # of validation loss for 10 consecutive epochs
    weights_path = f"./saved/models/test_{compiled_model.name}" + "_{epoch:02d}_{val_categorical_accuracy:.2f}.h5"
    checkpoint = ModelCheckpoint(
        weights_path,
        monitor='val_categorical_accuracy',
        verbose=1,
        save_best_only=True,

        # used if model subclass is used
        # save_weights_only=True,
        mode='max')
    stopper = EarlyStopping(monitor='val_categorical_accuracy', patience=10)
    callbacks = [checkpoint, stopper]

    # begin training test model
    history = compiled_model.fit(training_data,
        epochs=epochs,
        batch_size=batch_size, 
        callbacks=callbacks,
        validation_data=validation_data,
        verbose=2,)
    
    return history

class MOClassifierHyperModel(kt.HyperModel):
    def __init__(self, n_classes, name=None, tunable=True):
        super().__init__(name, tunable)
        self.n_classes = n_classes

    def build(self, hp):
        # convolutional layers hyper parameters
        hp_filter = hp.Choice('n_filter', values=[8, 16, 32, 64])
        hp_kernel_size = hp.Choice('kernel_size', values=[(3, 3), (4, 4), (5, 5)])
        hp_padding = hp.Choice('padding', values=['same', 'valid'])

        # pooling layers hyper parameters
        hp_pool_size = hp.Choice('pool_size', values=[(2, 2), (3, 3)])
        hp_pool_strides = hp.Choice('pool_strides', values=[(1, 1), (2, 2)])

        # fully connected layers hyper params
        hp_num_dense_units = hp.Choice('n_dense_units', values=[64, 32])

        # learning rate alpha
        hp_learning_rate = hp.Choice('learning_rate', values=[1.2, 0.03, 0.01, 0.0075, 0.003, 0.001])

        # regularization value lambda
        hp_lambda = hp.Choice('lambda', values=[1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.25, 0.125, 0.01])

        # the drop probability values, instead of keep probability
        hp_dropout = hp.Choice('dropout', values=[0.1, 0.2, 0.3, 0.4, 0.5])

         # define architecture
        model = Sequential(name='architecture-A')

        # build conv and poollayers
        model.add(Conv2D(filters=hp_filter, kernel_size=hp_kernel_size, strides=(1, 1), kernel_regularizer=L2(hp_lambda)))
        model.add(Activation(activation=tf.nn.relu))
        model.add(MaxPooling2D(pool_size=hp_pool_size, strides=hp_pool_strides, padding=hp_padding))
        model.add(Conv2D(filters=hp_filter, kernel_size=hp_kernel_size, strides=(1, 1), kernel_regularizer=L2(hp_lambda)))
        model.add(Activation(activation=tf.nn.relu))
        model.add(MaxPooling2D(pool_size=hp_pool_size, strides=hp_pool_strides, padding=hp_padding))

        # flatten pooled layers
        model.add(Flatten())

        # build fully connected layers
        model.add(Dense(units=hp_num_dense_units, kernel_regularizer=L2(hp_lambda)))
        model.add(BatchNormalization())

        # pass dense or batch normalized layer
        model.add(Activation(activation=tf.nn.relu))
        model.add(Dropout(rate=hp_dropout))
        model.add(Dense(units=hp_num_dense_units, kernel_regularizer=L2(hp_lambda)))
        model.add(BatchNormalization())
        model.add(Activation(activation=tf.nn.relu))
        model.add(Dropout(rate=hp_dropout))

        # add final dense layer with final dimension of
        # the dense_layers_dims value
        model.add(Dense(units=self.n_classes, kernel_regularizer=L2(hp_lambda)))

        # define loss, optimizer, and metrics
        loss = cce_loss(from_logits=True)
        opt = Adam(learning_rate=hp_learning_rate)
        metrics = [cce_metric(from_logits=True), CategoricalAccuracy()]
        model.compile(
            loss=loss,
            optimizer=opt,
            metrics=metrics
        )

        return model

def load_tuner(hyper_model, metric='val_categorical_accuracy', objective='max', max_epochs=10, factor=3, save_path: str='./saved/tuned_models'):
    
    obj = kt.Objective(metric, objective)

    tuner = kt.Hyperband(
        hyper_model, 
        objective=obj, 
        max_epochs=max_epochs,
        factor=factor,
        directory=save_path,
        project_name='tuned_models'
    )

    return tuner

def train_tuner(tuner, training_data, validation_data, epochs=10, batch_size=128):
    # define checkpoint and early stopping callback to save
    # best weights at each epoch and to stop if there is no 
    # improvement of validation loss for 3 consecutive epochs
    # since we are only using a tuner
    stopper = EarlyStopping(monitor='val_categorical_accuracy', patience=5)
    callbacks = [stopper]
    
    # fit model to data
    tuner.search(
        training_data,
        epochs=epochs, 
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks
    )

    best_params = tuner.get_best_hyperparameters()[0]

    return best_params

    
    
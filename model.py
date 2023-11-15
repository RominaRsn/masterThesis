import matplotlib.pyplot as plt
import numpy as np
import mne
import time
from keras.models import load_model

import tensorflow
from sklearn.model_selection import train_test_split
from masterThesis.metrics import metrics
import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from keras import layers,Sequential
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.constraints import max_norm

def Novel_CNN(input_size=(500, 1)):
    inputs = Input(input_size)
    conv1 = Conv1D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv1D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = AveragePooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = AveragePooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv1D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = AveragePooling1D(pool_size=2)(conv3)

    conv4 = Conv1D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv1D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = AveragePooling1D(pool_size=2)(drop4)

    conv5 = Conv1D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv1D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    pool5 = AveragePooling1D(pool_size=2)(drop5)

    conv6 = Conv1D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool5)
    conv6 = Conv1D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    drop6 = Dropout(0.5)(conv6)

    pool6 = AveragePooling1D(pool_size=2)(drop6)

    conv7 = Conv1D(2048, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool6)
    conv7 = Conv1D(2048, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    drop7 = Dropout(0.5)(conv7)

    flatten1 = Flatten()(drop7)
    output1 = Dense(500)(flatten1)  # Adjusted Dense layer output size
    model = Model(inputs=inputs, outputs=output1)

    #model.summary()
    return model

def simpleModel(input_shape=(500,1)):
    max_norm_value = 6.0
    model = Sequential()
    model.add(Conv1D(128, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
                     kernel_initializer='he_uniform', input_shape=input_shape))

    model.add(Conv1D(96, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
                     kernel_initializer='he_uniform'))
    model.add(
        Conv1DTranspose(96, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
                        kernel_initializer='he_uniform'))
    model.add(
        Conv1DTranspose(96, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
                        kernel_initializer='he_uniform'))
    model.add(
        Conv1D(1, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='tanh', padding='same'))

    model.summary()
    return model

def paper_Model(input_shape=(500, 1)):
    max_norm_value = 6.0
    model = Sequential()
    # Encoder
    model.add(Conv1D(128, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
                     kernel_initializer='he_uniform', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(96, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
                     kernel_initializer='he_uniform'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(96, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
                     kernel_initializer='he_uniform'))

    # Decoder
    model.add(Conv1DTranspose(96, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
                              kernel_initializer='he_uniform'))
    model.add(UpSampling1D(size=2))
    model.add(Conv1DTranspose(96, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
                              kernel_initializer='he_uniform'))
    model.add(UpSampling1D(size=2))
    model.add(Conv1DTranspose(96, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
                              kernel_initializer='he_uniform'))
    model.add(UpSampling1D(size=2))

    # Output layer
    model.add(
        Conv1D(1, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='sigmoid', padding='same'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

   # model.summary()
    return model


#a model with one more layer than simpleModel
def simpleModel_modified(input_shape=(500,1)):
    max_norm_value = 6.0
    model = Sequential()
    model.add(Conv1D(128, kernel_size=3, activation='relu',
                     kernel_initializer='he_uniform', input_shape=input_shape))

    model.add(Conv1D(96, kernel_size=3, activation='relu',
                     kernel_initializer='he_uniform'))
    model.add(Conv1D(64, kernel_size=3, activation='relu',
                     kernel_initializer='he_uniform'))  # Additional Conv1D

    model.add(
        Conv1DTranspose(64, kernel_size=3, activation='relu',
                        kernel_initializer='he_uniform'))
    model.add(
        Conv1DTranspose(128, kernel_size=3, activation='relu',
                                                             kernel_initializer='he_uniform'))
    model.add(
        Conv1DTranspose(96, kernel_size=3, activation='relu',
                                                             kernel_initializer='he_uniform'))
    model.add(
        Conv1D(1, kernel_size=3, activation='sigmoid', padding='same'))

    model.summary()
    return model

def simpleModel_modified2(input_shape=(500,1)):
    # Define the input layer
    input_layer = Input(shape=(500, 1))  # Assuming 1 channel (e.g., for time series data)

    # Encoding layers
    encoded1 = Conv1D(128, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(input_layer)
    encoded1 = MaxPooling1D(2, padding='same')(encoded1)

    encoded2 = Conv1D(128, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(encoded1)
    encoded2 = MaxPooling1D(2, padding='same')(encoded2)

    encoded3 = Conv1D(128, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(encoded2)
    encoded3 = MaxPooling1D(2, padding='same')(encoded3)

    # Decoding layers (symmetric to the encoding layers)
    decoded3 = Conv1D(128, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(encoded3)
    decoded3 = UpSampling1D(2)(decoded3)

    decoded2 = Conv1D(128, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(decoded3)
    decoded2 = UpSampling1D(2)(decoded2)

    decoded1 = Conv1D(128, 3, activation='relu',kernel_initializer='he_uniform', padding='valid')(decoded2)
    decoded1 = UpSampling1D(2)(decoded1)

    output_layer = Conv1D(1, 3, activation='tanh',kernel_initializer='he_uniform', padding='same')(decoded1)  # 1 channel for reconstruction

    # Create the autoencoder model
    autoencoder = Model(input_layer, output_layer)

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Print the summary of the autoencoder model
    autoencoder.summary()
    return autoencoder

def encoder_with_4_layers(input_shape=(500,1)):
    # Define the input layer
    input_layer = Input(shape=(500, 1))  # Assuming 1 channel (e.g., for time series data)

    # Encoding layers
    encoded1 = Conv1D(128, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(input_layer)
    encoded1 = MaxPooling1D(2, padding='same')(encoded1)

    encoded2 = Conv1D(128, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(encoded1)
    encoded2 = MaxPooling1D(2, padding='same')(encoded2)

    encoded3 = Conv1D(128, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(encoded2)
    encoded3 = MaxPooling1D(2, padding='same')(encoded3)

    encoded4 = Conv1D(128, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(encoded3)
    encoded4 = MaxPooling1D(2, padding='same')(encoded4)

    # Decoding layers (symmetric to the encoding layers)
    decoded4 = Conv1D(128, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(encoded4)
    decoded4 = UpSampling1D(2)(decoded4)

    decoded3 = Conv1D(128, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(encoded3)
    decoded3 = UpSampling1D(2)(decoded3)

    decoded2 = Conv1D(128, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(decoded3)
    decoded2 = UpSampling1D(2)(decoded2)

    decoded1 = Conv1D(128, 3, activation='relu',kernel_initializer='he_uniform', padding='valid')(decoded2)
    decoded1 = UpSampling1D(2)(decoded1)

    output_layer = Conv1D(1, 3, activation='tanh',kernel_initializer='he_uniform', padding='same')(decoded1)  # 1 channel for reconstruction

    # Create the autoencoder model
    autoencoder = Model(input_layer, output_layer)

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Print the summary of the autoencoder model
    autoencoder.summary()
    return autoencoder



def encoder_with_5_layers(input_shape=(500,1)):
    # Define the input layer
    input_layer = Input(shape=(500, 1))  # Assuming 1 channel (e.g., for time series data)

    # Encoding layers
    encoded1 = Conv1D(256, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(input_layer)
    encoded1 = MaxPooling1D(2, padding='same')(encoded1)

    encoded2 = Conv1D(256, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(encoded1)
    encoded2 = MaxPooling1D(2, padding='same')(encoded2)

    encoded3 = Conv1D(256, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(encoded2)
    encoded3 = MaxPooling1D(2, padding='same')(encoded3)

    encoded4 = Conv1D(256, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(encoded3)
    encoded4 = MaxPooling1D(2, padding='same')(encoded4)

    encoded5 = Conv1D(256, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(encoded4)
    encoded5 = MaxPooling1D(2, padding='same')(encoded5)

    # Decoding layers (symmetric to the encoding layers)
    decoded5 = Conv1D(256, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(encoded5)
    decoded5 = UpSampling1D(2)(decoded5)

    decoded4 = Conv1D(256, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(encoded4)
    decoded4 = UpSampling1D(2)(decoded4)

    decoded3 = Conv1D(256, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(encoded3)
    decoded3 = UpSampling1D(2)(decoded3)

    decoded2 = Conv1D(256, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(decoded3)
    decoded2 = UpSampling1D(2)(decoded2)

    decoded1 = Conv1D(256, 3, activation='relu',kernel_initializer='he_uniform', padding='valid')(decoded2)
    decoded1 = UpSampling1D(2)(decoded1)

    output_layer = Conv1D(1, 3, activation='tanh',kernel_initializer='he_uniform', padding='same')(decoded1)  # 1 channel for reconstruction

    # Create the autoencoder model
    autoencoder = Model(input_layer, output_layer)

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Print the summary of the autoencoder model
    autoencoder.summary()
    return autoencoder

encoder_with_5_layers()
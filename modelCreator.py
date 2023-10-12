import keras
from keras.models import Sequential, save_model
from keras.layers import Conv1D, Conv1DTranspose
from keras.constraints import max_norm
from scipy.signal import butter,filtfilt,iirnotch
max_norm_value = 6
input_shape = (500, 1)


model = Sequential()
model.add(Conv1D(128, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
                         kernel_initializer='he_uniform', input_shape=input_shape))
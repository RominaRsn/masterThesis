from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import LeakyReLU, Add
from tensorflow.keras import layers,Sequential
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf


def LSTM_net():
    datanum = 500
    model = tf.keras.Sequential()

    #Input layer
    model.add(Input(shape=(datanum, 1)))

    #LSTM layers
    model.add(LSTM(80, return_sequences=True))
    model.add(Activation('tanh'))
    model.add(LSTM(80, return_sequences=True))
    model.add(Activation('tanh'))
    model.add(LSTM(80, return_sequences=True))
    model.add(Activation('tanh'))
    model.add(LSTM(80, return_sequences=True))
    model.add(Activation('tanh'))
    model.add(LSTM(80, return_sequences=True))
    model.add(Activation('tanh'))

    #Output layer
    model.add(TimeDistributed(Dense(units=1)))

    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Create the LSTM network
# lstm_model = LSTM_net()




# def GRU_net():
    # datanum = 500
    # model = tf.keras.Sequential()
    # model.add(Input(shape=(datanum, 1)))
    # model.add(layers.GRU(35, return_sequences=True))
    # model.add(layers.Activation('tanh'))
    # model.add(layers.GRU(29, return_sequences=True))
    # model.add(layers.Activation('tanh'))
    # model.add(layers.GRU(35, return_sequences=True))
    # model.add(layers.Activation('tanh'))

    # model.add(layers.TimeDistributed(layers.Dense(units=1)))
    # model.summary()
    # model.compile(optimizer='adam', loss='mean_squared_error')
    # return model


model = LSTM_net()
#model = load_model( r'C:\Users\Romina\masterThesis\trainedModel\gru_model_EMG_EOG.keras')


data_clean_normalized = np.load(r"C:\Users\Romina\Downloads\clean_normalized_new.npy")
data_noisy_normalized = np.load(r"C:\Users\Romina\Downloads\noisy_normalized_new.npy")


noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized, data_clean_normalized, test_size=0.2, random_state=42)


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
checkpoint_path = r'C:\Users\Romina\masterThesis\trainedModel\lstm_model_EMG.keras'
checkpoint = ModelCheckpoint(checkpoint_path,
                             monitor='val_loss',  # You can choose a different metric, e.g., 'val_accuracy'
                             save_best_only=True,  # Save only if the validation performance improves
                             mode='min',  # 'min' for loss, 'max' for accuracy, 'auto' will infer automatically
                             verbose=1)
model.optimizer.learning_rate = 1e-3
model.fit(
    noisy_train,
    clean_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    callbacks=[checkpoint, callback],
    shuffle=True
)


del noisy_train, noisy_test, clean_train, clean_test, data_clean_normalized, data_noisy_normalized

# model = GRU_net()


data_clean_eog = np.load(r"C:\Users\Romina\Downloads\clean_data_eog_normalized.npy")
data_noisy_eog = np.load(r"C:\Users\Romina\Downloads\noisy_data_eog_normalized.npy")



# Step 1: Split into training and test sets
#noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized_cheby, data_clean_normalized_cheby, test_size=0.2, random_state=42)

noisy_train_eog, noisy_test_eog, clean_train_eog, clean_test_eog = train_test_split(data_noisy_eog, data_clean_eog, test_size=0.2, random_state=42)

model = load_model(r'C:\Users\Romina\masterThesis\trainedModel\lstm_model_EMG.keras')

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)

checkpoint_path = r'C:\Users\Romina\masterThesis\trainedModel\lstm_model_EMG_EOG.keras'

checkpoint = ModelCheckpoint(checkpoint_path,
                             monitor='val_loss',  # You can choose a different metric, e.g., 'val_accuracy'
                             save_best_only=True,  # Save only if the validation performance improves
                             mode='min',  # 'min' for loss, 'max' for accuracy, 'auto' will infer automatically
                             verbose=1)

model.optimizer.learning_rate = 1e-3
model.fit(
    noisy_train_eog,
    clean_train_eog,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    callbacks=[callback, checkpoint],
    shuffle=True
)

result = model.predict(noisy_test_eog)

np.save(r'C:\Users\Romina\masterThesis\trainedModel\result_gru_eog.npy', result)

#todo calculate the result of EMG
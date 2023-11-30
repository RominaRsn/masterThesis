from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from scipy.signal import butter,filtfilt,iirnotch
from masterThesis.metrics import metrics
from keras.utils import plot_model
import os
import masterThesis.model as models
import tensorflow as tf


data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_new.npy")
data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_new.npy")




# Step 1: Split into training and test sets
noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized, data_clean_normalized, test_size=0.2, random_state=42)


smaller_noisy_train = noisy_train[0:1000]
smaller_clean_train = clean_train[0:1000]


model = models.deep_CNN()

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)

model.optimizer.learning_rate = 1e-6
model.fit(
    noisy_train,
    noisy_test,
    epochs=2,
    batch_size=16,
    validation_split=0.1,
    callbacks=[callback],
    shuffle=True
)

model.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\deep_CNN.h5")

result = model.predict(noisy_test)
np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\results\deep_CNN_result.npy", result)

# plt.plot(result[10, :,:])
# plt.plot(clean_train[10, :])
# plt.plot(noisy_train[10, :])
# plt.legend(['result', 'clean', 'noisy'])
#
# plt.show()
#
#
# plt.plot(result[100, :,:])
# plt.plot(clean_train[100, :])
# plt.plot(noisy_train[100, :])
# plt.legend(['result', 'clean', 'noisy'])
#
# plt.show()
#
#
# plt.plot(result[100, :,:])
# plt.plot(clean_train[100, :])
# plt.plot(noisy_train[100, :])
# plt.legend(['result', 'clean', 'noisy'])
#
# plt.show()
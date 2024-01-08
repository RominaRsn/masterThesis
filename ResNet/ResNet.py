import tensorflow as tf
from keras import layers, models, optimizers
from keras.applications import ResNet50
import numpy as np
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt


# Assume your X and Y data are numpy arrays
# X: Input signals
# Y: Regression targets

# Load a pretrained ResNet50 model without the top (classification) layer

# Freeze the layers of the pretrained ResNet (optional)



# data_clean_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_resNet.npy")
# data_noisy_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_resNet.npy")
#
# print("data loaded")

expanded_clean = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_resNet_expanded.npy")
expanded_noisy = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_resNet_expanded.npy")


noisy_train, noisy_test, clean_train, clean_test = train_test_split(expanded_noisy, expanded_clean, test_size=0.2, random_state=42)


#
# print("data loaded")
# print(expanded_clean.shape)
# print(expanded_noisy.shape)
#
# print("training started")
#
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
#
# for layer in base_model.layers:
#     layer.trainable = False


# # Reshape your input data to have three channels (channels-last format)
# X_reshaped = data_noisy_normalized_cheby.reshape(data_noisy_normalized_cheby.shape[0], 32, 32, 1)  # Adjust the reshape based on your data
# Y_reshaped = data_clean_normalized_cheby.reshape(data_clean_normalized_cheby.shape[0], 32, 32, 1)  # Adjust the reshape based on your data

# Add your regression head on top of the ResNet
# model = models.Sequential()
# model.add(base_model)
# model.add(layers.GlobalAveragePooling2D())  # Global average pooling for 1D signal
# model.add(layers.Dense(1, activation='linear'))  # Assuming you have a single regression output
#
# # Compile the model
# model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')





# Train the model with your data
#model.fit(noisy_train, clean_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[ModelCheckpoint('resnet_checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')])
model = load_model(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\ResNet\resnet_checkpoint.h5')

result = model.predict(noisy_test[0:1])

r_ch1 = result[0][:, :, 0]
r_ch2 = result[0][:, :, 1]
r_ch3 = result[0][:, :, 2]

c_ch1 = clean_test[0][:, :, 0]
n_ch1 = noisy_test[0][:, :, 0]

r_ch1 = r_ch1.reshape(32, 32)
r_ch2 = r_ch2.reshape(32, 32)
r_ch3 = r_ch3.reshape(32, 32)

c_ch1 = c_ch1.reshape(32, 32)
n_ch1 = n_ch1.reshape(32, 32)

r_ch1_flat = r_ch1.flatten()
r_ch2_flat = r_ch2.flatten()
r_ch3_flat = r_ch3.flatten()

c_ch1_flat = c_ch1.flatten()
n_ch1_flat = n_ch1.flatten()


fig, axes = plt.subplots(nrows=7, ncols=1, sharey='col')

axes[0].plot(c_ch1_flat, label = 'Clean Data')
axes[0].set_title('Clean data')
axes[0].set_ylabel('Signal amplitude')
axes[0].set_xlabel('Time')

#print(smaller_reshaped_data_clean_test[row_index, :].shape)


axes[1].plot(n_ch1_flat, label = 'Noisy Data')
axes[1].set_title('Noisy data')
axes[1].set_ylabel('Signal amplitude')
axes[1].set_xlabel('Time')

#result = model.predict(result)
#result = result.transpose()

# axes[2].plot(np.convolve(result_ae[row_index, :], result_classic[row_index, :], mode='full'), label='predicted data- all skip connections')
# #axes[2].plot(sharpedened_result_1[row_index, :], label='sharpened')
# axes[2].set_title('predicted data - all skip connections')
# axes[2].set_ylabel('Signal amplitude')
# axes[2].set_xlabel('Time')
# axes[2].legend(loc='lower right')

#
axes[3].plot(r_ch1_flat, label ='channel 1')
#axes[3].plot(sharpedened_result_2[row_index, :], label='sharpened')
axes[3].set_title('channel 1')
axes[3].set_ylabel('Signal amplitude')
axes[3].set_xlabel('Time')
axes[3].legend(loc='lower right')

axes[4].plot(r_ch2_flat, label ='channel 2')
axes[4].set_title('channel 2')
axes[4].set_ylabel('Signal amplitude')
axes[4].set_xlabel('Time')
axes[4].legend(loc='lower right')

axes[5].plot(r_ch3_flat, label ='channel 3')
axes[5].set_title('channel 3')
axes[5].set_ylabel('Signal amplitude')
axes[5].set_xlabel('Time')
axes[5].legend(loc='lower right')

plt.show()
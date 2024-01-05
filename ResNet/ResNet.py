import tensorflow as tf
from keras import layers, models, optimizers
from keras.applications import ResNet50
import numpy as np
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.models import load_model

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



print("data loaded")
print(expanded_clean.shape)
print(expanded_noisy.shape)

print("training started")

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

for layer in base_model.layers:
    layer.trainable = False


# # Reshape your input data to have three channels (channels-last format)
# X_reshaped = data_noisy_normalized_cheby.reshape(data_noisy_normalized_cheby.shape[0], 32, 32, 1)  # Adjust the reshape based on your data
# Y_reshaped = data_clean_normalized_cheby.reshape(data_clean_normalized_cheby.shape[0], 32, 32, 1)  # Adjust the reshape based on your data

# Add your regression head on top of the ResNet
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())  # Global average pooling for 1D signal
model.add(layers.Dense(1, activation='linear'))  # Assuming you have a single regression output

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')





# Train the model with your data
model.fit(noisy_train, clean_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[ModelCheckpoint('resnet_checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')])
model = load_model('resnet_checkpoint.h5')
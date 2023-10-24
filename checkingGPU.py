import tensorflow as tf

# Check if GPU is available
print(tf.config.experimental.list_physical_devices('GPU'))

# Run on GPU if available, otherwise use CPU
device = '/GPU:0' if tf.config.experimental.list_physical_devices('GPU') else '/CPU:0'
print("Using device:", device)
import tensorflow as tf

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check for available devices
devices = tf.config.list_physical_devices()
print("Available devices:")
for device in devices:
    print(device)

# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    print("GPU is available and being used by TensorFlow.")
else:
    print("GPU is not available, TensorFlow is using the CPU.")
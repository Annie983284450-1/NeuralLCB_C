import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.config.list_physical_devices('GPU'):
    print("Your GPU is detected: ", tf.config.list_physical_devices('GPU'))
else:
    print("GPU not detected")
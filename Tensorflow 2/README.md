## Tensorflow 2
This directory contains the Tensorflow 2.0.0 alpha equivalent of the main code files. 
tf.layers and tf.keras.layers have been unified. Keras functional API is now much easier to work with. Tensorflow specific function calls(like tf.nn.leaky_relu()) do not break the Keras model if not wrapped in a keras lambda layer, anymore.

import numpy as np
import tensorflow as tf

_LEAKY_RELU = 0.1
_BATCH_NORM_EPSILON = 1e-05


class Resnet18(object):
    def conv2d_block(self,inputs, filters, kernel_size, strides=1):
        inputs = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(inputs)
        inputs = tf.keras.layers.BatchNormalization(epsilon=_BATCH_NORM_EPSILON,trainable=True)(inputs)
        inputs = tf.keras.layers.Lambda(tf.nn.leaky_relu,
                                arguments={'alpha':_LEAKY_RELU})(inputs)
        return inputs

    def residual_block(self,inputs,filters,strides=1):
        shortcut = inputs
        shortcut = tf.keras.layers.Conv2D(filters,1,strides=strides)(shortcut)
        
        inputs = self.conv2d_block(inputs,filters,3,strides=strides)
        inputs = self.conv2d_block(inputs, filters,3)
        inputs = tf.keras.layers.Dropout(0.3)(inputs)
        
        inputs = tf.keras.layers.Add()([inputs,shortcut])
        return inputs
    
    def build_model(self,inputs):
        ip = inputs
        with tf.variable_scope('conv_1'):
            inputs = self.conv2d_block(inputs,64,7,strides=2)
        
        with tf.variable_scope('conv_2'):
            inputs = self.conv2d_block(inputs,64,3,strides=2)
            inputs = self.residual_block(inputs,64)
            inputs = self.residual_block(inputs,64)

        with tf.variable_scope('conv_3'):
            inputs = self.residual_block(inputs,128,strides=2)
            inputs = self.residual_block(inputs,128)

        with tf.variable_scope('conv_4'):
            inputs = self.residual_block(inputs,256,strides=2)
            inputs = self.residual_block(inputs,256)

        with tf.variable_scope('conv_5'):
            inputs = self.residual_block(inputs,512,strides=2)
            inputs = self.residual_block(inputs,512)

        #Avg-Pool
        inputs = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        outputs = tf.keras.layers.Dense(4,activation=tf.keras.activations.sigmoid)(inputs)
        # print(outputs)
        return tf.keras.Model(inputs=ip,outputs=outputs,name='resnet-18')
        # return logits
    

class KerasResnet18(object) :
    def __new__(cls,inputs):
        print("KERAS MODEL CREATED")
        model = Resnet18()
        outputs = tf.keras.layers.Lambda(model.forward_pass)(inputs)
        return tf.keras.Model(inputs=inputs,outputs=outputs)


    
    


import tensorflow as tf
import numpy as np
from resnet18 import KerasResnet18,Resnet18
import pandas as pd
import keras
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers


inputs = tf.keras.Input((416,416,3))
model = Resnet18().build_model(inputs)

test_df = pd.read_csv('test.csv')
# print(test_df.head())
datagen = ImageDataGenerator()
test_generator = datagen.flow_from_dataframe(test_df,directory='../Dataset/images_test/',
                    x_col = 'image_name',
                    y_col = None,
                    target_size=(416,416),batch_size=1,class_mode=None,shuffle=False)

# def eval_loss(y_true,y_pred):
#     # print(y_true.shape)
#     mask = tf.ones_like(y_true)
#     mask[:,:2] *= 640
#     mask[:,2:] *= 480
#     true = tf.multiply(y_true,(np.array([640,640,480,480])*mask)
#     pred = tf.multiply(y_pred,tf.convert_to_tensor(mask))
#     return tf.reduce_mean(tf.abs(true-pred))

model.load_weights('weights_52-1.000-0.794.h5')
model.compile(optimizer='adam',loss="binary_crossentropy")
# print(model.summary())

out = model.predict_generator(test_generator,steps=len(test_generator),use_multiprocessing=True,
                        workers=8,verbose=1)

out = np.array(out)

# Scale predictions to actual image size
out[:,:2] *= 640
out[:,2:] *= 480
out = np.array(out,dtype=np.int32)
test_df['x1'] = out[:,0]
test_df['x2'] = out[:,1]
test_df['y1'] = out[:,2]
test_df['y2'] = out[:,3]

test_df.to_csv('Final.csv',index=False)
# model.evaluate_generator(test_generator,steps=10,verbose=1)

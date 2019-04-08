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

valid_df = pd.read_csv('test.csv')
# print(len(valid_df))
# valid_df['image_name'] = '/home/atom/common_data/Projects/FlipkartGRID/Fastai/images/' + valid_df['image_name']
print(valid_df.head())
datagen = ImageDataGenerator()
valid_generator = datagen.flow_from_dataframe(valid_df,directory='Fastai/images_test/',
                    x_col = 'image_name',
                    # y_col = ['x1','x2','y1','y2'],
                    y_col = None,
                    target_size=(416,416),batch_size=1,class_mode=None,shuffle=False)
# print(next(valid_generator)[1])

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
# with open('model_architecture.json', 'w') as f:
#     f.write(model.to_json())
# print(model.summary())

out = model.predict_generator(valid_generator,steps=len(valid_generator),use_multiprocessing=True,
                        workers=8,verbose=1)

out = np.array(out)
out[:,:2] *= 640
out[:,2:] *= 480
out = np.array(out,dtype=np.int32)
valid_df['x1'] = out[:,0]
valid_df['x2'] = out[:,1]
valid_df['y1'] = out[:,2]
valid_df['y2'] = out[:,3]

valid_df.to_csv('Submission.csv',index=False)
# model.evaluate_generator(valid_generator,steps=10,verbose=1)


# with tf.Session() as sess:
#     out = sess.run(tf.nn.sigmoid(out))





import tensorflow as tf
import numpy as np
from resnet18_tf_v2 import KerasResnet18,Resnet18
import pandas as pd
import keras
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import save_model
from tensorflow.keras import optimizers


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.backend.tensorflow_backend import set_session

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
tf.keras.backend.set_session(session)



def loss(y_true,y_pred):
    # l = tf.square(tf.sqrt(tf.abs(y_pred))-tf.sqrt(tf.abs(y_true)))
    # l = 5*tf.reduce_mean(l)


    l = tf.math.squared_difference(y_pred,y_true)
    l = tf.reduce_mean(input_tensor=l)
    l = 5*tf.sqrt(l)
    return l

# def loss_iou(y_true,y_pred):
# def loss_iou(y_true,y_pred):
#     print(y_pred.shape)
#     print(tf.minimum(y_true[:,0],y_pred[:,0]).shape)
#     pt1 = tf.concat([tf.reshape(tf.minimum(y_true[:,0],y_pred[:,0]),(-1,1)),tf.reshape(tf.minimum(y_true[:,2],y_pred[:,2]),(-1,1))],axis=-1)
#     pt2 = tf.concat([tf.reshape(tf.minimum(y_true[:,1],y_pred[:,1]),(-1,1)),tf.reshape(tf.minimum(y_true[:,3],y_pred[:,3]),(-1,1))],axis=-1)
#     pt3 = tf.concat([tf.reshape(tf.maximum(y_true[:,0],y_pred[:,0]),(-1,1)),tf.reshape(tf.maximum(y_true[:,2],y_pred[:,2]),(-1,1))],axis=-1)
#     pt4 = tf.concat([tf.reshape(tf.maximum(y_true[:,1],y_pred[:,1]),(-1,1)),tf.reshape(tf.maximum(y_true[:,3],y_pred[:,3]),(-1,1))],axis=-1) 
#     print(pt3.shape)
#     intersection = tf.reduce_prod(tf.abs(pt3-pt2),axis=-1)
#     square1 = tf.reduce_prod(tf.abs(pt1-pt2),axis=-1)
#     square2 = tf.reduce_prod(tf.abs(pt3-pt4),axis=-1)
#     print(square1.shape)
#     union = tf.abs(square1+square2-intersection)
#     iou = tf.reduce_mean(intersection/union)
# #     print(iou)

#     l = tf.squared_difference(y_pred,y_true)
#     l = tf.reduce_mean(l)
#     l = 5*tf.sqrt(l)

#     return 2*(1-iou)+l


    


inputs = tf.keras.Input((416,416,3))
train_df = pd.read_csv('Train_Set.csv')
valid_df = pd.read_csv('Valid_Set.csv')
datagen = ImageDataGenerator()
train_generator = datagen.flow_from_dataframe(train_df,directory='../Dataset/images',
                    x_col = 'image_name',y_col = ['x1','x2','y1','y2'],
                    target_size=(416,416),batch_size=4,class_mode='other')
valid_generator = datagen.flow_from_dataframe(valid_df,directory='..Dataset/images',
                    x_col = 'image_name',y_col = ['x1','x2','y1','y2'],
                    target_size=(416,416),batch_size=4,class_mode='other')

# print(next(train_generator)[0].shape)

# valid_x,valid_y=[],[]
# for x,y in valid_generator:
#     valid_x.append(x)
#     valid_y.append(y)

# valid_generator = datagen.flow(x=x,y=y,batch_size=4)


model = Resnet18().build_model(inputs)
adam = optimizers.Adam(lr=9e-5)
# model.load_weights('weights_290_0.116.h5')
# model.load_weights('Model.h5')
model.compile(optimizer=adam,loss=loss,metrics=['accuracy'])

checkpoint = ModelCheckpoint('weights_{epoch:02d}_{val_loss:.3f}.h5',
                                monitor='val_loss',save_best_only=True,
                                save_weights_only=True)
# early = EarlyStopping(monitor='val_acc',patience=20)
board = TensorBoard(log_dir='logs/',histogram_freq=0,batch_size=4)
red = ReduceLROnPlateau(monitor='val_loss',patience=5,min_delta=3e-3,factor=0.5,min_lr=5e-6)
callbacks_list = [checkpoint,board,red]        

# model.fit(images,labels,epochs=200,batch_size=4,validation_split=.01,
#         callbacks=callbacks_list,steps_per_epoch=200)
print(model.summary())
model.fit_generator(generator=train_generator,steps_per_epoch=200,
                    validation_data=valid_generator,validation_steps=8,
                    epochs=550,callbacks=callbacks_list,
                    use_multiprocessing=False,workers=0,initial_epoch=0)
save_model(model,'IOU_Model.h5')
# model.fit_generator(generator=train_generator,steps_per_epoch=200,
#                     validation_data=valid_generator,validation_steps=10,
#                     epochs=100,callbacks=callbacks_list,
#                     use_multiprocessing=True,initial_epoch=50)
# save_model(model,'Model.h5')
# model.fit_generator(generator=train_generator,steps_per_epoch=200,
#                     validation_data=valid_generator,validation_steps=10,
#                     epochs=150,callbacks=callbacks_list,
#                     use_multiprocessing=True,initial_epoch=100)
# save_model(model,'Model.h5')
# model.fit_generator(generator=train_generator,steps_per_epoch=200,
#                     validation_data=valid_generator,validation_steps=10,
#                     epochs=200,callbacks=callbacks_list,
#                     use_multiprocessing=True,initial_epoch=150)
# save_model(model,'Model.h5')


import tensorflow as tf
import numpy as np
from resnet18 import Resnet18
import pandas as pd
import keras
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import save_model
from tensorflow.keras import optimizers

def loss(y_true,y_pred):
    l = tf.squared_difference(y_pred,y_true)
    l = tf.reduce_mean(l)
    l = 5*tf.sqrt(l)
    return l    

def main():
    inputs = tf.keras.Input((416,416,3))
    train_df = pd.read_csv('Train_Set.csv')
    valid_df = pd.read_csv('Valid_Set.csv')
    datagen = ImageDataGenerator()
    train_generator = datagen.flow_from_dataframe(train_df,directory='../Dataset/images',
                        x_col = 'image_name',y_col = ['x1','x2','y1','y2'],
                        target_size=(416,416),batch_size=4,class_mode='other')
    valid_generator = datagen.flow_from_dataframe(valid_df,directory='../Dataset/images',
                        x_col = 'image_name',y_col = ['x1','x2','y1','y2'],
                        target_size=(416,416),batch_size=4,class_mode='other')


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
    
    print(model.summary())
    model.fit_generator(generator=train_generator,steps_per_epoch=200,
                        validation_data=valid_generator,validation_steps=8,
                        epochs=550,callbacks=callbacks_list,
                        use_multiprocessing=False,workers=0,initial_epoch=0)
    save_model(model,'IOU_Model.h5')

if __name__ == '__main__':
    main()

import os
import sys
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


def U_Net():
    """
    Basic U-Net architecture with strided convolutions as pooling layers. Loosely adapted from Guo et al. https://arxiv.org/abs/1807.04686.
    """
    inpt = Input(shape=(None, None, 1)) 
    
    #four layers conv2d + relu, size 64
    x1 = Conv2D(filters=64, kernel_size=(3, 3), activation = 'relu', padding='same', strides=(1,1))(inpt)
    for i in range(3):
        x1 = Conv2D(filters=64, kernel_size=(3, 3), activation = 'relu', padding='same', strides=(1,1))(x1)
    
    #pooling
    pool1 = Conv2D(filters=256, kernel_size=(2, 2), padding='valid', strides=(2, 2))(x1)
    y1 = Conv2D(filters=128, kernel_size=(1, 1), padding='valid', strides=(1, 1))(pool1)
    #three layers conv2d + relu, size 128
    for i in range(3):
        y1 = Conv2D(filters=128, kernel_size=(3, 3), activation = 'relu', padding='same', strides=(1,1))(y1)
    
    #pooling
    pool2 = Conv2D(filters=512, kernel_size=(2, 2), padding='valid', strides=(2, 2))(y1)
    z1 = Conv2D(filters=256, kernel_size=(1, 1), padding='valid', strides=(1, 1))(pool2)
    #three layers conv2d + relu, size 256
    for i in range(3):
        z1 = Conv2D(filters=256, kernel_size=(3, 3), activation = 'relu', padding='same', strides=(1,1))(z1)

    #pooling
    pool3 = Conv2D(filters=1024, kernel_size=(2, 2), padding='valid', strides=(2, 2))(z1)
    a1 = Conv2D(filters=512, kernel_size=(1, 1), padding='valid', strides=(1, 1))(pool3)
    #three layers conv2d + relu, size 256
    for i in range(3):
        a1 = Conv2D(filters=512, kernel_size=(3, 3), activation = 'relu', padding='same', strides=(1,1))(a1)

    #pooling
    pool4 = Conv2D(filters=2048, kernel_size=(2, 2), padding='valid', strides=(2, 2))(a1)
    b = Conv2D(filters=1024, kernel_size=(1, 1), padding='valid', strides=(1, 1))(pool4)
    #seven layers conv2d + relu, size 1024
    for i in range(6):
        b = Conv2D(filters=1024, kernel_size=(3, 3), activation = 'relu', padding='same', strides=(1,1))(b)
    b = Conv2D(filters=2048, kernel_size=(3, 3), activation = 'relu', padding='same', strides=(1,1))(b)

    #upsample + sum
    up0 = Conv2DTranspose(filters=512, kernel_size=(2, 2), padding='valid', strides=(2, 2), name='Transpose1')(b)
    sum0 = Add()([up0, a1])

    #three layers size 512    #UNet:

    a2 = Conv2D(filters=512, kernel_size=(3, 3), activation = 'relu', padding='same', strides=(1,1))(sum0)
    a2 = Conv2D(filters=512, kernel_size=(3, 3), activation = 'relu', padding='same', strides=(1,1))(a2)
    a2 = Conv2D(filters=1024, kernel_size=(3, 3), activation = 'relu', padding='same', strides=(1,1))(a2)
    
    #upsample + sum
    up1 = Conv2DTranspose(filters=256, kernel_size=(2, 2), padding='valid', strides=(2, 2), name='Transpose2')(a2)
    sum1 = Add()([up1, z1])

    #three layers size 512
    z2 = Conv2D(filters=256, kernel_size=(3, 3), activation = 'relu', padding='same', strides=(1,1))(sum1)
    z2 = Conv2D(filters=256, kernel_size=(3, 3), activation = 'relu', padding='same', strides=(1,1))(z2)
    z2 = Conv2D(filters=512, kernel_size=(3, 3), activation = 'relu', padding='same', strides=(1,1))(z2)
    
    #upsample + sum
    up2 = Conv2DTranspose(filters=128, kernel_size=(2, 2), padding='valid', strides=(2, 2), name='Transpose3')(z2)
    sum2 = Add()([up2, y1])

    #three layers size 128
    y2 = Conv2D(filters=128, kernel_size=(3, 3), activation = 'relu', padding='same', strides=(1,1))(sum2)
    y2 = Conv2D(filters=128, kernel_size=(3, 3), activation = 'relu', padding='same', strides=(1,1))(y2)
    y2 = Conv2D(filters=256, kernel_size=(3, 3), activation = 'relu', padding='same', strides=(1,1))(y2)
    
    #upsample + sum
    up3 = Conv2DTranspose(filters=64, kernel_size=(2, 2), padding='valid', strides=(2, 2), name='Transpose4')(y2)
    sum3 = Add()([up3, x1])

    #three layers size 64
    x2 = Conv2D(filters=64, kernel_size=(3, 3), activation = 'relu', padding='same', strides=(1,1))(sum3)
    x2 = Conv2D(filters=64, kernel_size=(3, 3), activation = 'relu', padding='same', strides=(1,1))(x2)
    x2 = Conv2D(filters=1, kernel_size=(3, 3), padding='same', strides=(1,1))(x2)

    img_out = Add(name='img_out')([inpt, x2]) 
    model = Model(inputs=inpt, outputs=img_out)
    return model

def effnet(): 
    """
    EfficientNetB2 pretrained on the ImageNet Dataset. Last denselayer substituted with denselayer with six outputs.
    """
    model_base = tf.keras.applications.EfficientNetB2(weights='imagenet', include_top=False, pooling='avg')
    inpt = Input([None, None, 3])
    x = model_base(inpt)
    x = Dropout(name='top_dropout', rate=0.3)(x)
    x = Dense(name='predictions', units=6, activation='sigmoid', kernel_initializer='VarianceScaling')(x)
    model = Model(inputs=inpt, outputs=x)
    return model



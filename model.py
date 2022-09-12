# This model is just a code snipet from the jupyter notebook. User needs to initialize x_train and y_train on its own.
# Its highly recommended to use jupyter notebook where entire data analysis is shown. 

import tensorflow as tf
import keras
import segmentation_models as sm

backbone = 'resnet34'
preprocess_input = sm.get_preprocessing(backbone)

model = sm.Unet(backbone, encoder_weights = 'imagenet')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])

history = model.fit(x_train, y_train, batch_size = 8, epochs= 10, verbose=1, validation_data=(x_val, y_val))
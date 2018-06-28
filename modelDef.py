import numpy as np
from utils import *
from keras.layers import Input, Dense, Activation, Conv2D, Conv3D, Reshape, Lambda, RNN, LSTM, Flatten, TimeDistributed
from keras.models import Sequential, Model
from keras.layers import advanced_activations, pooling, Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend as K

############################################################################################
###################---------------- PARAMETERS DEFINTION ----------------###################
############################################################################################

loss = ['categorical_crossentropy']
# loss = ['binary_crossentropy']
last_activation = 'softmax'
# last_activation = 'sigmoid'
metrics = ['categorical_accuracy']
# metrics = ['binary_accuracy']
optimizer = 'nadam'
nOut = int(fftSize)+1
# nOut = int(fftSize)
input_shape = [int(fftSize), int(timeDepth), int(nHarmonics)]

model = Sequential()
model.add(TimeDistributed(BatchNormalization(), input_shape=[int(batchSize), int(fftSize), int(timeDepth), int(nHarmonics)]))
model.add(TimeDistributed(Conv2D(128, (5, 1), padding='same', activation='sigmoid', data_format="channels_last")))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Conv2D(128, (1, 12), padding='same', activation='sigmoid', data_format="channels_last")))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Conv2D(1, (1, 1), padding='same', activation='sigmoid', data_format="channels_last")))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dense(nOut, activation='softmax')))
model.add(LSTM(nOut, activation='sigmoid', stateful=False, return_sequences=True))
model.add(TimeDistributed(Dense(nOut, activation=last_activation)))
model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
return model, False

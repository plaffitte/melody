import google.protobuf

#import keras, tensorflow
#from keras.models import Sequential, Model
#from keras.layers import Dense, Activation, Conv2D, Conv3D, Input, Convolution2D
#import numpy as np
#import random, os, glob
#import mir_eval
#import evaluate
#from predict_on_audio import load_model


#
# fftSize = 360
# timeDepth = 10
# nharmonics = 10
#
# # Define the shape of filters in the following order:
# #   - frequency
# #   - time
# #   - harmonics
# filterShape1 = [1, 1, 10]
# filterShape2 = [1, 1, 8]
# filterShape3 = [1, 1, 6]
# filterShape4 = [1, 1, 4]
#
# featureMaps = 32 # Define the number of feature maps
# input_size = [fftSize, timeDepth, nharmonics, 1] # Shape of input to conv network
# inputs = Input(input_size)
# batchSize = 128
# dataN = 1000
#
# # --->>> Model creation method 1
# # myModel = Sequential()
# # layer = keras.layers.Conv2D(filters=featureMaps1, kernel_size=filterShape2, strides=1, padding='same', activation='sigmoid', input_shape=input_size)
# # myModel.add(layer)
#
# # --->>> Model creation method 2
# layer1 = Conv3D(featureMaps, filterShape1, padding='valid', activation='sigmoid')(inputs)
# # layer1Flat = keras.layers.Flatten()(layer1)
# layer2 = Conv3D(featureMaps, filterShape2, padding='valid', activation='sigmoid')(inputs)
# # layer2Flat = keras.layers.Flatten()(layer2)
# layer3 = Conv3D(featureMaps, filterShape3, padding='valid', activation='sigmoid')(inputs)
# # layer3Flat = keras.layers.Flatten()(layer3)
# layer4 = Conv3D(featureMaps, filterShape4, padding='valid', activation='sigmoid')(inputs)
# # layer4Flat = keras.layers.Flatten()(layer4)
# concatLayer = keras.layers.concatenate([layer1, layer2, layer3, layer4], axis=3)
# reduceDim = Conv3D(1, (10, 10, 10), padding='valid', activation='sigmoid')(concatLayer)
# flatLayer = keras.layers.Flatten()(reduceDim)
# outputFin = Dense(fftSize, activation='sigmoid')(flatLayer)
#
# myModel = Model(inputs, outputFin)
#
# myModel.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# # # Train the model, iterating on the data in batches of 32 samples
# # myModel.fit(data, labels, epochs=10, batch_size=batchSize)
#
# for i in range(int(np.floor(dataN / batchSize))):
#     data = np.random.random((batchSize, fftSize, timeDepth, nharmonics, 1))
#     labels = np.random.uniform(0, 1, size=(batchSize, fftSize))
#     myModel.train_on_batch(data, labels)
#
# print myModel.summary()

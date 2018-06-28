# import keras
# from keras.models import Sequential, Model
# from keras.layers import Dense, Dropout, Conv2D, advanced_activations, pooling, Flatten
import os, os.path, glob, sys
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
import medleydb as mdb
from random import shuffle
from utils import binarize, generateDummy, log
from model import model
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import keras
from keras.layers import Input, Dense, Activation, Conv2D, Conv3D, Reshape, Lambda, RNN, LSTM, Flatten, TimeDistributed, GRU, SimpleRNN
from keras.models import Sequential, Model
from keras.layers import advanced_activations, pooling, Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.datasets import mnist
from keras.optimizers import RMSprop

### ---------------------------------------------------------------------------------------------------------
### ---------------------------------------------------------------------------------------------------------
# N = 10000 # size of dataset
# S = 128 # batch size
# nEpochs = 10
#
# model = Sequential ()
#
# """Conv 64x3x3 leaky relu activation"""
# model.add(Conv2D(64, 3, 3, border_mode='valid',
#                         batch_input_shape=(None, 115,80,1),
#                         init = 'orthogonal'))
# model.add(advanced_activations.LeakyReLU(alpha=0.01))
#
# """Conv 32x3x3 leaky relu activation"""
# model.add(Conv2D(32, 3, 3, border_mode='valid',
#             init = 'orthogonal'))
# model.add(advanced_activations.LeakyReLU(alpha=0.01))
# """Pooling 3x3"""
# model.add(pooling.MaxPooling2D(pool_size=(3, 3),
#                                dim_ordering='default'))
#
#
# """Conv 128x3x3 leaky relu activation"""
# model.add(Conv2D(128, 3, 3, border_mode='valid',
#                         init='orthogonal'))
# model.add(advanced_activations.LeakyReLU(alpha=0.01))
#
# """Conv 64x3x3 leaky relu activation"""
# model.add(Conv2D(64, 3, 3, border_mode='valid',
#                         init='orthogonal'))
#
# model.add(advanced_activations.LeakyReLU(alpha=0.01))
# """Pooling 3x3"""
# model.add(pooling.MaxPooling2D(pool_size=(3, 3)))
#
# model.add(Flatten())
#
# """FC 256 units leaky relu activation"""
# model.add(Dropout(0.5))
# model.add(Dense(256,
#             init='he_normal'))
# model.add(advanced_activations.LeakyReLU(alpha=0.01))
# """FC 64 unitsleaky relu activation"""
#
# model.add(Dropout(0.5))
# model.add(Dense(64,
#                 init='he_normal'))
# model.add(advanced_activations.LeakyReLU(alpha=0.01))
# """FC 1 unit sigmoid activation"""
# model.add(Dropout(0.5))
# model.add(Dense(1, input_dim = 64, activation='sigmoid',
#                 init='he_normal'))
#
# opt = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999,
#                                   epsilon=1e-08, decay=0.0)
#
# model.compile(loss='binary_crossentropy', optimizer=opt)
#
# data = np.random.random((10000, 115, 80, 1))
# labels = np.random.uniform(0, 1, size=(10000, 1))
#
# def dataGen(S):
#     while 1:
#         for i in range(int(np.floor(len(data) / S))):
#             yield [data[i*S : i*S + S, :, :, :], labels[i*S:i*S+S]]
#
# dataGenerator = dataGen(S)
# steps = N / S
# # Train the model, iterating on the data in batches of 32 samples
# # model.fit(data, labels, epochs=10, batch_size=128)
# model.fit_generator(dataGenerator, steps_per_epoch=steps , epochs=nEpochs)
#
# print model.summary()
#
### ---------------------------------------------------------------------------------------------------------
### ---------------------------------------------------------------------------------------------------------

# # fs, audio = wav.read('/u/anasynth/laffitte/MedleyDB/Audio/AClassicEducation_NightOwl/AClassicEducation_NightOwl_MIX.wav')
# # i = 2000
# # fig, ax = plt.subplots(1)
# # ax.plot(np.linspace(0, 100*i-1, 100*i), audio[0:100*i,0])
# # # ax.plot(np.linspace(0, 3*i-1, 3*i), audio[0:3*i,0], c='b')
# # # ax.plot(np.linspace(3*i, 5*i-1, 2*i), audio[3*i:5*i,0], c='r')
# # # ax.plot(np.linspace(5*i, 9*i-1, 4*i), audio[5*i:9*i,0], c='b')
# # # ax.plot(np.linspace(9*i, 11*i-1, 2*i), audio[9*i:11*i,0], c='r')
# # # ax.plot(np.linspace(11*i, 19*i-1, 8*i), audio[11*i:19*i,0], c='b')
# # ax.set_ylim([-20000,20000])
# # plt.show()
#
### ---------------------------------------------------------------------------------------------------------
### ---------------------------------------------------------------------------------------------------------

# class fake(object):
#
#     def __init__(self, id):
#         name = "fake"
#         id = np.ones(id)
#
# def generator():
#     for i in range(10):
#         print("I am generator 1")
#         yield fake(i)
#
# def imbricatedGenerator(generator1):
#     i = 0
#     for gen in generator1:
#         print("here is my generator: ", gen.name)
#         for k in gen.id:
#             print("I am generator 2")
#             yield k
#         i += 1
#
# def generator_test():
#
#     gen1 = generator()
#     gen2 = imbricatedGenerator(gen1)
#
#     for k in range(2):
#         gen3 = gen1
#         for i in gen3:
#             # print(i)
#             pass
#         print("Epcoh:", k)
#
#     print("Done!")
#
# # if __name__=="__main__":
# #     print("Running generator test program")
# #     generator_test()

### ---------------------------------------------------------------------------------------------------------
### ---------------------------------------------------------------------------------------------------------

# def predictOnBatch(ind):
#     dataBatch = np.zeros((1,1024,72,10,6))
#     for k in range(1024):
#         if ind*1024+10*k+10 <= dataBlock.shape[2]:
#             dataBatch[0,k,:,:,:]=dataBlock[:,:,ind*1024+10*k:ind*1024+10*k+10].transpose(1,2,0)
#
#     plt.subplot(3,1,1)
#     plt.imshow(dataBatch[0,:,:,0,0].T, aspect='auto')
#     pred = model.predict(dataBatch)
#     return pred
#
# modelPath='/data/Experiments/1D-VOICING/weights.49-0.64.hdf5'
# model=keras.models.load_model(modelPath)
# dataPath='/data/Experiments/1D-VOICING/features/'
# dataFile=os.listdir(dataPath)
# targetPath='/data/Experiments/1D-VOICING/targets/'
# targetFile=os.listdir(targetPath)
# targetFile=sorted(targetFile)
# dataFile=sorted(dataFile)
# print "File Name:", dataFile[1]
# dataBlock=np.load(os.path.join(dataPath,dataFile[1]))
# labelBlock=np.load(os.path.join(targetPath,targetFile[1]))
#
# plt.imshow(labelBlock, aspect='auto')
# plt.show()
#
# N = 10
# for i in range(N):
#     pred = predictOnBatch(i)
#     plt.subplot(3,1,2)
#     plt.imshow(labelBlock[:,i*1024:i*1024+1024],aspect='auto')
#     plt.subplot(3,1,3)
#     plt.imshow(pred[0].T,aspect='auto')
#     plt.show()

# # define our time-distributed setup
# inp = Input(shape=(8, 28, 28, 1))
# x = TimeDistributed(Conv2D(8, (4, 4), padding='valid', activation='relu'))(inp)
# x = TimeDistributed(Conv2D(16, (4, 4), padding='valid', activation='relu'))(x)
# x = TimeDistributed(Flatten())(x)
# x = GRU(units=100, return_sequences=True)(x)
# # x = GRU(units=50, return_sequences=False)(x)
# # x = Dropout(.2)(x)
# # x = Dense(1)(x)
# model = Model(inp, x)
#
# rmsprop = RMSprop()
# model.compile(loss='mean_squared_error', optimizer=rmsprop)
# print model.summary()

### ---------------------------------------------------------------------------------------------------------
### ---------------------------------------------------------------------------------------------------------
### -------------------------------   TEST RNN MODEL WITH DUMMY DATA    -------------------------------------
# log('')
# log('---> Training Statefull Model <---')
# batchSize = 100
# timeDepth = 20
# nHarmonics = 6
# hopSize = 1
# nEpochs = 50
# fftSize = 72
# myModel, modelSplit = model("1D-CATEGORICAL-statefull", batchSize, fftSize, timeDepth, nHarmonics, False, True, 1, True)
# print(myModel.summary())
# meanTrainAccuracy = []
# meanTrainLoss = []
# meanValidLoss = []
# meanValidAccuracy = []
# count = 0
# patience = 5
# data = None
# labels = None
# for epoch in range(nEpochs):
#     if count >= patience:
#         break
#     log("\n Training Epoch {}".format(epoch))
#     sys.stdout.flush()
#     # Get each song in batches and train
#     for i in range(1500):
#         batch, targets = generateDummy(1, batchSize, [timeDepth, nHarmonics], fftSize)
#         # if newSong:
#         #     myModel.reset_states()
#         trainLoss, trainAccuracy = myModel.train_on_batch(batch, targets)
#         meanTrainLoss.append(trainLoss)
#         meanTrainAccuracy.append(trainAccuracy)
#         if data is None:
#             data = batch[0,:,:,10,0]
#             labels = targets[0,:,:]
#         else:
#             data = np.concatenate((data, batch[0,:,:,10,0]))
#             labels = np.concatenate((labels, targets[0,:,:]))
#     fig, (ax1, ax2) = plt.subplots(2,1,1)
#     ax1.imshow(data)
#     ax1.imshow(labels)
#     plt.plot()
#     # Validate model with validation dataset
#     for i in range(100):
#         batch, targets = generateDummy(1, batchSize, [timeDepth, nHarmonics], fftSize)
#         # if newSong:
#         #     myModel.reset_states()
#         validLoss, validAccuracy = myModel.test_on_batch(batch, targets)
#         meanValidLoss.append(validLoss)
#         meanValidAccuracy.append(validAccuracy)
#     log(("Training Loss: {} <--> Accuracy: {}").format(np.mean(meanTrainLoss), np.mean(meanTrainAccuracy)))
#     log(("Validation Loss: {} <--> Accuracy: {}").format(np.mean(meanValidLoss), np.mean(meanValidAccuracy)))
#     if epoch==0:
#         myModel.save(os.path.join('/data/Experiments/Test', "weights.{}-{}.h5".format(epoch, np.mean(meanValidLoss))))
#         prevLoss = np.mean(meanValidLoss)
#     elif np.mean(meanValidLoss) < prevLoss:
#         myModel.save(os.path.join('/data/Experiments/Test', "weights.{}-{}".format(epoch, np.mean(meanValidLoss))))
#         prevLoss = np.mean(meanValidLoss)
#         if count>0:
#             count = 0
#     else:
#         count += 1

### ---------------------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------------------
from predict_on_audio import load_model
import matplotlib.pyplot as plt
path = '/data3/Data/360*60/features'
files = os.listdir(path)
data = np.load(os.path.join(path,files[0]))
print "Shape of data:", data.shape
length = data.shape[-1]
timeDepth = 50
fftSize = 360
batchSize = 10
nharmonics = 6
L = length/timeDepth
N = L/batchSize
deepModel = load_model('melody2')
out = None
for i in range(N):
    batch = np.zeros((batchSize, timeDepth, fftSize, nharmonics))
    for ii in range(batchSize):
        batch[ii] = data[:,:,ii*timeDepth:ii*timeDepth + timeDepth].transpose(2,1,0)
    if out is None:
        out = deepModel.predict(batch)
    else:
        out = np.concatenate((out, deepModel.predict(batch)))
print "Shape of output:", out.shape
pred = None
for k in range(out.shape[0]):
    if pred is None:
        pred = out[k]
    else:
        pred = np.concatenate((pred, out[k]))
plt.imshow(pred.T, aspect='auto')
plt.show()

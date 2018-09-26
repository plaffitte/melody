import os, sys, pandas, glob, re, csv
from data_creation import DataSet, createAnnotation, readAnnotation, getLabels, load_model, getLabelMatrix
import medleydb as mdb
import numpy as np
import matplotlib.pyplot as plt
from utils import log
import keras
import librosa
import mir_eval
from model import model
from evaluation import plotScores, writeScores
from evaluation import get_time_grid, get_freq_grid, pitch_activations_to_melody, calculate_metrics, writeScores

#######################################################################################################################################################################################################################
'''                                             MODEL TEST
 '''
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

#######################################################################################################################################################################################################################
'''                                     GENERATOR Test
'''

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
########################################################################################################################################################################################################################
'''                             TEST RNN MODEL with DUMMY DATA
'''
# log('')
# log('---> Training Statefull Model <---')
# batchSize = 500
# timeDepth = 50
# nHarmonics = 6
# hopSize = 50
# nEpochs = 100
# fftSize = 360
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
########################################################################################################################################################################################################################
'''                             GENERATE INPUT DATA FROM GENERATOR AND COMPUTE SCORES
                                        ON DEEP SALIENCE REPRESENTATION
'''
# outPath = '/u/anasynth/laffitte/test'
# dataPath = '/u/anasynth/laffitte/test'
# inputPath = os.path.join(dataPath, 'features')
# targetPath = os.path.join(dataPath, 'targets')
# modelDim="BASELINE"
# params = {}
# params['batchSize']=1000
# params['stateFull']="False"
# params['timeDepth']=50
# params['nHarmonics']=6
# params['hopSize']=50
# dataobj = DataSet(inputPath, targetPath, modelDim)
# fftSize = 360
# binsPerOctave = 60
# nOctave = 6
# rnnBatch = 32
# nHarmonics = 6
# trainSet, validSet, testSet = dataobj.partDataset()
# testSet = testSet[:2]
# dataobj.getFeature(testSet, modelDim, outPath, 60, 6, nHarmonics, False)
# # myModel=keras.models.load_model(os.path.join(outPath, 'weights.04-0.78.hdf5'))
# train = False
# voicing = False
#
# ### RETRIEVE LABELS AND INPUTS
# labels, preds = getLabels('TOTO', dataobj, testSet, params, modelDim, voicing, fftSize, rnnBatch)
# batchSize = int(params['batchSize'])
# bucketList = dataobj.bucketDataset(testSet, rnnBatch)
# predictions = []
# targets = []
# all_scores = []
#
# all_scores, melodyEstimation, _, _ = calculate_metrics(dataobj, preds, labels, preds, testSet, int(binsPerOctave), int(nOctave), outPath, batchSize, rnnBatch, thresh=0.05, voicing=False)
# print(all_scores)
# writeScores(all_scores, outPath)
#
# ### PLOT PREDS/INPUTS AND LABELS FOR EACH SONG ###
# cmap = 'viridis'
# for [prediction, target, song] in zip(predictions, targets, testSet):
#     # prediction = binarize(prediction)
#     print("song:", song)
#     if target is not None:
#         if np.shape(target)[0] > np.shape(target)[1]:
#             target = target.T
#         if np.shape(prediction)[0] > np.shape(prediction)[1]:
#             prediction = prediction.T
#         if isinstance(song, list):
#             song = song[0]
#         # fig, (ax1, ax2, ax3) = plt.subplots(3,1)
#         fig, (ax1, ax2) = plt.subplots(2,1)
#         ax1.imshow(prediction, aspect='auto', cmap=cmap, vmax=1, vmin=0, origin='lower')
#         ax1.set_title('INPUTS')
#         ax2.imshow(target, interpolation='nearest', aspect='auto', cmap=cmap, vmax=1, vmin=0, origin='lower')
#         ax2.set_title('TARGETS')
#         plt.savefig(os.path.join(outPath, song+'_result_melody1.png'))
#         plt.close()
########################################################################################################################################################################################################################
'''                         COMPUTE SCORES for AClassicEducation_NightOwl
                        directly with DEEPSALIENCEREPRESENTATION via RACHEL's CODE
'''
# binsPerOctave = 60
# nOctave = 6
# HOP_LENGTH = 256
# Fs = 22050
# FMIN = 32.7
# voicing = False
# thresh = 0.05
# ### LOAD INPUT FEATURES AS NUMPY ARRAY ###
# inputFile = '/data/anasynth_nonbp/laffitte/Experiments/deepSalienceRepresentations/features/AClassicEducation_NightOwl_mel2_input.npy'
# inputs = np.load(inputFile)
# idx = np.array(np.arange(0,inputs.shape[-1],44100/Fs), dtype='int32')
# inputs = inputs[:,:,idx]
#
# ### CREATE TARGET FROM CSV ANNOTATIONS ###
# annotFile = '/data2/anasynth_nonbp/laffitte/MedleyDB/Annotations/Melody_Annotations/MELODY2/AClassicEducation_NightOwl_MELODY2.csv'
# targetFile = pandas.read_csv(annotFile, sep=',')
# data = readAnnotation(annotFile)
# annot = np.asarray(np.array(data).T[0])
# annot.reshape((np.shape(annot)[0], np.shape(annot)[1]))
# times = annot[:,0]
# freqs = annot[:,1]
# freq_grid = librosa.cqt_frequencies(binsPerOctave*nOctave, FMIN, bins_per_octave=binsPerOctave)
# length = int(np.floor(len(times)*(Fs/44100)))
# time_grid = librosa.core.frames_to_time(range(int(length)), sr=Fs, hop_length=HOP_LENGTH)
# target = createAnnotation(freq_grid, time_grid, times, freqs)
#
# ### GET PREDICTIONS ###
# input_hcqt = inputs.transpose(1, 2, 0)[np.newaxis, :, :, :]
# n_t = input_hcqt.shape[2]
# n_slices = 200
# t_slices = list(np.arange(0, n_t, n_slices))
# output_list = []
# deepModel = load_model('melody2')
# for i, t in enumerate(t_slices):
#     prediction = deepModel.predict(input_hcqt[:, :, t:t+n_slices, :], verbose=0)
#     output_list.append(prediction[0, :, :])
# predicted_output = np.hstack(output_list)
# pred = predicted_output.T
#
# ## COMPUTE SCORES ###
# scores = calculate_metrics(None, [pred[:14000,:]], [target.T[:14000,:]], [inputs[:,:14000]], ['AClassicEducation_NightOwl'], binsPerOctave, nOctave, '/u/anasynth/laffitte/test/1', 1000, 16, thresh=0.05, voicing=True)
# print(scores)
########################################################################################################################################################################################################################
'''                                     COMPARE BINARY TARGET WITH ANNOTATION
'''
# binsPerOctave = 60
# nOctave = 6
# HOP_LENGTH = 256
# Fs = 22050
# FMIN = 32.7
# voicing = False
# thresh = 0.38
#
# ### Create TARGET from CSV ANNOTATIONS ###
# annotFile = '/data2/anasynth_nonbp/laffitte/MedleyDB/Annotations/Melody_Annotations/MELODY2/Auctioneer_OurFutureFaces_MELODY2.csv'
# targetFile = pandas.read_csv(annotFile, sep=',')
# data = readAnnotation(annotFile)
# annot = np.asarray(np.array(data).T[0])
# annot.reshape((np.shape(annot)[0], np.shape(annot)[1]))
# times = annot[:,0]
# freqs = annot[:,1]
# freq_grid = librosa.cqt_frequencies(
#     binsPerOctave*nOctave, FMIN, bins_per_octave=binsPerOctave
#     )
# length = int(np.floor(len(times)*(Fs/44100)))
# time_grid = librosa.core.frames_to_time(
#     range(int(length)), sr=Fs, hop_length=HOP_LENGTH
# )
# target = createAnnotation(freq_grid, time_grid, times, freqs)
#
# ### Re-Create ANNOTATION MELODY from TARGET
# ref_times = get_time_grid(binsPerOctave, nOctave, len(target), Fs, 256)
# ref_freqs = pitch_activations_to_melody(
# target.T, binsPerOctave, nOctave, thresh=thresh, voicing=voicing, mod=False
# )
#
# idx = np.array(np.arange(0,len(ref_freqs)*2,44100/Fs), dtype='int32')
# plt.plot(freqs[idx]-ref_freqs)
# plt.show()
# plt.close()
 ######################################################################################################################################################################################################################
'''                              TEST INPUT FORMATTING, ENCODING AND DECODING
                                by visualizing Data directly after formatting
'''
# if os.path.isdir('/data2/anasynth_nonbp/laffitte'):
#     audioPath = '/data2/anasynth_nonbp/laffitte/MedleyDB/Audio/'
#     annotPath = '/data2/anasynth_nonbp/laffitte/MedleyDB/Annotations/Melody_Annotations/MELODY2'
# else:
#     annotPath = '/net/as-sdb/data/mir2/MedleyDB/Annotations/Melody_Annotations/MELODY2'
#     audioPath = '/net/as-sdb/data/mir2/MedleyDB/Audio/'
# targetDim = "BASELINE"
# batchSize = 1000
# stateFull = "False"
# fftSize = 360
# binsPerOctave = 60
# nOctave = 6
# rnnBatch = 5
# nHarmonics = 6
# voicing = False
# timeDepth = 50
# Fs = 22050
# hopSize = 50
# stateFull = True if stateFull=="True" else False
# params = {}
# params['batchSize']= batchSize
# params['stateFull']= stateFull
# params['timeDepth']= timeDepth
# params['nHarmonics']= nHarmonics
# params['hopSize']= hopSize
# realTestSet = []
# outPath = '/u/anasynth/laffitte/test/outputs'
# path = '/u/anasynth/laffitte/test'
# if not os.path.isdir(outPath):
#     os.mkdir(outPath)
# inputPath = os.path.join(path, 'features')
# targetPath = os.path.join(path, 'targets')
# dataObj = DataSet(inputPath, targetPath, targetDim)
# trainSet, validSet, testSet = dataObj.partDataset()
# testSet = testSet[0:9]
# dirList = sorted(os.listdir(audioPath))
# dirList = [i for i in dirList if '.' not in i]
# for j in dirList:
#     fileList = sorted(os.listdir(os.path.join(audioPath, j)))
#     fileList = [k[:-8] for k in fileList if re.match('[^._].*?.wav', k)]
#     tracks = [tr for tr in fileList if tr in testSet]
#     for i in tracks:
#         annotFile = os.path.join(annotPath, i+'_MELODY2.csv')
#         if annotFile is not None and os.path.exists(annotFile):
#             realTestSet.extend(tracks)
# dataObj.getFeature(realTestSet, targetDim, path, 60, 6, nHarmonics, False)
# labels, inputs, trackList = getLabels('TOTO', dataObj, realTestSet, params, targetDim, voicing, fftSize, rnnBatch)
# labels, inputs, trackList = getLabelMatrix('TOTO', dataObj, realTestSet, params, targetDim, voicing, fftSize, rnnBatch)
# inp = []
# predictions = []
# targets = []
# offset = 0
# bucketList = dataObj.bucketDataset(trackList, rnnBatch)
# for (b, subTracks) in enumerate(bucketList):
#     L, longest = dataObj.findLongest(subTracks[0])
#     nSequences = int(np.floor(L/batchSize))
#     for s, song in enumerate(sorted(subTracks[0])):
#         inp.append(None)
#         predictions.append(None)
#         targets.append(None)
#         for k in np.arange(s, nSequences*rnnBatch, rnnBatch):
#             pred = inputs[offset + k]
#             truth = labels[offset + k]
#             curInput = inputs[offset + k]
#             if truth is not None and np.array(truth.nonzero()).any():
#                 # mask = np.where(truth==-1)
#                 # if any(mask[1]):
#                 #     curInput = curInput[0:mask[0][0],:]
#                 #     truth = truth[0:mask[0][0],:]
#                 #     pred = pred[0:mask[0][0],:]
#                 if truth.any():
#                     if inp[-1] is None:
#                         inp[-1] = curInput
#                         predictions[-1] = pred
#                         targets[-1] = truth
#                     else:
#                         inp[-1] = np.concatenate((inp[-1], curInput))
#                         predictions[-1] = np.concatenate((predictions[-1], pred))
#                         targets[-1] = np.concatenate((targets[-1], truth))
#         j = glob.glob(os.path.join(targetPath, '{}_mel2_target.npy'.format(song)))
#         if j:
#             curTarget = np.load(j[0])
#         fig, (ax1, ax2) = plt.subplots(2,1)
#         ax1.imshow(curTarget, aspect='auto', origin='lower')
#         ax2.imshow(targets[-1].T, aspect='auto', origin='lower')
#         plt.savefig(os.path.join(path, song))
#         plt.close()
#     offset += k+1

# mysong = labels
# inp = inputs
# scores, _, _, _ = calculate_metrics(dataObj, inp, mysong, inp, trackList, binsPerOctave, nOctave, '/u/anasynth/laffitte/test/', batchSize, rnnBatch, thresh=0.05, voicing=True)
# print(scores)
# writeScores(scores, outPath)

 ######################################################################################################################################################################################################################
'''                              COMPUTE SCORES ON PREDICTIONS
'''

preds = np.load('')
inp = np.load('')
mysong = np.load('')

scores,_,_,_ = calculate_metrics(dataObj, preds, mysong, inp, trackList, binsPerOctave, nOctave, '/u/anasynth/laffitte/test/', batchSize, rnnBatch, thresh=0.05, voicing=True)

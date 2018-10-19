import numpy as np
import keras, librosa
import matplotlib.pyplot as plt
import os, path, re, csv
from predict_on_audio import load_model
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import GRU, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, TimeDistributed
import json, sklearn
import medleydb as mdb
from data_creation import getLabelMatrix

'''                     RUN A SIMPLE RNN ON RACHEL'S predictions
'''
# Fs = 22050
# nOut = 361
# seqLength = 500
# trainingDir = '/u/anasynth/laffitte/test'
# deepModel = load_model('melody2')
# deepModel.predict(np.zeros((2,nOut,50,6)))
# path = '/u/anasynth/laffitte/test'
# HARMONICS = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#
# def getFeature(dataSet, outPath, binsPerOctave, nOctave, nHarmonics=1):
#
#     print("--> Creating features \n")
#     if os.path.isdir('/data2/anasynth_nonbp/laffitte'):
#         audioPath = '/data2/anasynth_nonbp/laffitte/MedleyDB/Audio/'
#         annotPath = '/data2/anasynth_nonbp/laffitte/MedleyDB/Annotations/Melody_Annotations/MELODY2'
#     else:
#         annotPath = '/net/as-sdb/data/mir2/MedleyDB/Annotations/Melody_Annotations/MELODY2'
#         audioPath = '/net/as-sdb/data/mir2/MedleyDB/Audio/'
#     dirList = sorted(os.listdir(audioPath))
#     dirList=[i for i in dirList if '.' not in i]
#     H = []
#     for j in dirList:  # loop over all audio files
#         fileList = sorted(os.listdir(os.path.join(audioPath, j)))
#         fileList = [k[:-8] for k in fileList if re.match('[^._].*?.wav', k)]
#         tracks = [tr for tr in fileList if tr in dataSet]
#         for i in tracks:
#             prefix = i.replace('_MIX.wav', '')
#             inFile = os.path.join(outPath, '{}_mel2_input.npy'.format(prefix))
#             outFile = os.path.join(outPath, '{}_mel2_target.npy'.format(prefix))
#             if not os.path.exists(inFile) or not os.path.exists(outFile):
#                 audioFile = os.path.join(audioPath, os.path.join(j, i+'_MIX.wav'))
#                 annotFile = os.path.join(annotPath, i+'_MELODY2.csv')
#                 if annotFile is not None and os.path.exists(annotFile):
#                     ### Compute CQT
#                     if not os.path.exists(inFile):
#                         H = computeHcqt(audioFile, nHarmonics, binsPerOctave, nOctave)
#                         np.save(inFile, H.astype(np.float32))
#                     if not os.path.exists(outFile):
#                         ### Get labels
#                         data = readAnnotation(annotFile)
#                         annot = np.asarray(np.array(data).T[0])
#                         annot.reshape((np.shape(annot)[0], np.shape(annot)[1]))
#                         times = annot[:,0]
#                         freqs = annot[:,1]
#                         freq_grid = librosa.cqt_frequencies(
#                             binsPerOctave*nOctave, 32.7, bins_per_octave=binsPerOctave
#                             )
#                         length = int(np.floor(len(times)*(Fs/44100)))
#                         time_grid = librosa.core.frames_to_time(
#                             range(int(length)), sr=Fs, hop_length=256
#                         )
#                         target = createAnnotation(freq_grid, time_grid, times, freqs)
#                         np.save(outFile, target.astype(np.float32))
#
# def split(trackList):
#
#     data_splits_path = '/net/inavouable/u.anasynth/laffitte/Code/ismir2017-deepsalience/outputs/data_splits.json'
#     with open(data_splits_path, 'r') as fhandle:
#         data_splits = json.load(fhandle)
#     ### Get test set from Rachel's paper data
#     testSet = data_splits['test']
#     restSet = [tr for tr in trackList if tr not in testSet]
#     ### GET TRAIN AND VALID DATASET FROM THE REST OF MedleyDB DATASET
#     trainSet, validSet = sklearn.model_selection.train_test_split(restSet, train_size=0.75)
#     return trainSet, validSet, testSet
#
# def Generator(dataset, trainingDir):
#
#     while 1:
#         for song in dataset:
#             if os.path.exists(os.path.join(trainingDir, '{}_mel2_target.npy'.format(song))):
#                 feat = np.load(os.path.join(trainingDir, '{}_mel2_input.npy'.format(song)))
#                 annot = np.load(os.path.join(trainingDir, '{}_mel2_target.npy'.format(song)))
#                 length = feat.shape[-1]
#                 for k in range(int(np.floor(length/seqLength))):
#                     if k*seqLength+seqLength < length:
#                         inputsCnn = feat[:,:,k*seqLength:k*seqLength+seqLength].transpose(1,2,0)[None,:,:,:]
#                         deepSalience = deepModel.predict(inputsCnn)
#                         inputs = deepSalience[0,:,:].transpose(1,0)[None,:,:]
#                         tar = annot[:,k*seqLength:k*seqLength+seqLength].transpose(1,0)
#                         # targets = computeVoicing(targets)[None,:,:]
#                         targets = 0 * np.ones((seqLength, tar.shape[1]+1))
#                         for m in range(tar.shape[0]):
#                             targets[m, :] = computeVoicing(tar[m, :])
#                     yield inputs, targets[None,:,:]
#
# def computeVoicing(activations):
#
#     # import pdb; pdb.set_trace()
#     if len(activations.shape)==1:
#         ind = np.where((activations==0),0,0)
#         voicingArray = activations
#         voicingArray = np.insert(voicingArray, -1, 0)
#         if not any(activations):
#             voicingArray[ind] = 1
#     else:
#         for k in range(activations.shape[0]):
#             ind = np.where((activations[k,:]==0),0,0)
#             voicingArray = activations
#             voicingArray = np.insert(voicingArray, -1, 0)
#             if not any(activations[k,:]):
#                 voicingArray[ind] = 1
#
#     return voicingArray
#
# def lenDataset(dataset, trainingDir):
#     L = 0
#     for song in dataset:
#         if os.path.exists(os.path.join(trainingDir, '{}_mel2_target.npy'.format(song))):
#             annot = np.load(os.path.join(trainingDir, '{}_mel2_target.npy'.format(song)))
#             length = annot.shape[-1]
#             L += int(np.floor(length/seqLength))
#     return L
#
# def readAnnotation(filePath, num_cols=None, header=False):
#
#     if filePath is not None and os.path.exists(filePath):
#         with open(filePath) as f_handle:
#             annotation = []
#             linereader = csv.reader(f_handle)
#
#             # skip the headers for non csv files
#             if header:
#                 header = next(linereader)
#             else:
#                 header = []
#
#             for line in linereader:
#                 if num_cols:
#                     line = line[:num_cols]
#                 annotation.append([float(val) for val in line])
#         return annotation, header
#     else:
#         return None, None
#
# def createAnnotation(freq_grid, time_grid, annotation_times,
#                              annotation_freqs):
#     """Create the binary annotation target labels
#     """
#
#     n_freqs = len(freq_grid)
#     n_times = len(time_grid)
#     idx = np.array(np.arange(0, len(annotation_times), 44100/Fs), dtype='int32')
#     annotation_times = annotation_times[idx]
#     annotation_freqs = annotation_freqs[idx]
#
#     # time_bins = grid_to_bins(time_grid, 0.0, time_grid[-1])
#     time_bins = time_grid
#     freq_bins = grid_to_bins(freq_grid, 0.0, freq_grid[-1])
#     annot_time_idx = np.digitize(annotation_times, time_bins) - 1
#     annot_freq_idx = np.digitize(annotation_freqs, freq_bins) - 1
#
#     annotation_target = np.zeros((n_freqs+1, n_times))
#     annotation_target[annot_freq_idx, annot_time_idx] = 1
#
#     return annotation_target[1:,:]
#
# def grid_to_bins(grid, start_bin_val, end_bin_val):
#     """Compute the bin numbers from a given grid
#     """
#     bin_centers = (grid[1:] + grid[:-1])/2.0
#     bins = np.concatenate([[start_bin_val], bin_centers, [end_bin_val]])
#     return bins
#
# def findLongest(dataset):
#
#     longest = 0
#     name = ''
#     songDic = {}
#     for track in dataset:
#         if os.path.exists(os.path.join(trainingDir, '{}_mel2_target.npy'.format(track))):
#             j = np.load(os.path.join(trainingDir, '{}_mel2_target.npy'.format(track)))
#             if j:
#                 curTarget = np.load(j[0])
#                 if curTarget.shape[-1] >= longest:
#                     longest = curTarget.shape[-1]
#                     name = track
#                 songDic[track] = curTarget.shape[-1]
#     return longest, name
#
# def bucketDataset(dataset, size):
#
#     sortedList = []
#     bucketList = []
#     dataList = dataset[:]
#     for k in range(len(dataset)):
#         L, longest = findLongest(dataList)
#         if L != 0:
#             sortedList.append(longest)
#             dataList.remove(longest)
#     for k in range(int(np.floor(len(dataset)/size))+1):
#         bucketList.append([])
#         bucketList[-1].append(sortedList[k*size: k*size+size])
#
#     return bucketList
#
# def computeHcqt(audio_fpath, nHarmonics, binsPerOctave, nOctave):
#
#     y, fs = librosa.load(audio_fpath, sr=Fs)
#     cqt_list = []
#     shapes = []
#     harmonics = HARMONICS[0:nHarmonics]
#     for h in harmonics:
#         cqt = librosa.cqt(
#             y, sr=Fs, hop_length=256, fmin=32.7*float(h),
#             n_bins=binsPerOctave*nOctave,
#             bins_per_octave=binsPerOctave
#         )
#         cqt_list.append(cqt)
#         shapes.append(cqt.shape)
#
#     shapes_equal = [s == shapes[0] for s in shapes]
#     if not all(shapes_equal):
#         min_time = np.min([s[1] for s in shapes])
#         new_cqt_list = []
#         for i in range(len(cqt_list)):
#             new_cqt_list.append(cqt_list[i][:, :min_time])
#         cqt_list = new_cqt_list
#
#     log_hcqt = ((1.0/80.0) * librosa.core.amplitude_to_db(
#         np.abs(np.array(cqt_list)), ref=np.max)) + 1.0
#
#     return log_hcqt
#
# trackList = mdb.TRACK_LIST_V1
# trainSet, validSet, testSet = split(trackList)
# # getFeature(trainSet, path, 60, 6, 6)
# dataGenerator = Generator(trainSet, trainingDir)
# validationGenerator = Generator(validSet, trainingDir)
# trainSteps = lenDataset(trainSet, trainingDir)
# validSteps = lenDataset(validSet, trainingDir)
# testSteps = lenDataset(testSet, trainingDir)
# myModel = Sequential()
# myModel.add(TimeDistributed(BatchNormalization(), input_shape=[seqLength, 360]))
# myModel.add(Bidirectional(GRU(256, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)))
# myModel.add(TimeDistributed(Dense(nOut, activation='softmax')))
# myModel.compile(loss=['categorical_crossentropy'], metrics=['categorical_accuracy'], optimizer='adam')
# print(myModel.summary())
# filepath = os.path.join(path, "weights.{epoch:02d}-{loss:.2f}.hdf5")
# myModel.fit_generator(
#     generator=dataGenerator,
#     steps_per_epoch=trainSteps,
#     epochs=int(100),
#     validation_data=validationGenerator,
#     validation_steps=validSteps,
#     callbacks=[
#     keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, verbose=1),
#     keras.callbacks.ReduceLROnPlateau(patience=10),
#     keras.callbacks.EarlyStopping(patience=20, mode='min'),
#     ],
#     verbose = 2,
#     shuffle = False
#     )
# params = {}
# params['batchSize'] = seqLength
# params['timeDepth'] = seqLength
# params['nHarmonics'] = 6
# params['hopSize'] = 50
# params['nEpochs'] = 1
# params['stateFull']="False"
# # myModel = keras.models.load_model('/u/anasynth/laffitte/test/weights.09-2.52.hdf5')
#
# batchSize = int(params['batchSize'])
# timeDepth = int(params['timeDepth'])
# nHarmonics = int(params['nHarmonics'])
# hopSize = int(params['hopSize'])
# nEpochs = int(params['nEpochs'])
# stateFull = True if params['stateFull']=="True" else False
# myModel.summary()
# predictGenerator = Generator(testSet, trainingDir)
# preds = myModel.predict_generator(predictGenerator, steps=testSteps, verbose=1)
#
# trackList = []
# bucketList = dataobj.bucketDataset(testSet, 1)
# for (s, subTracks) in enumerate(bucketList):
#     for (k, track) in enumerate(sorted(subTracks[0])):
#         trackList.append(track)
# labels = None
# inputs = None
# for l in range(size):
#     one, two = gen.__next__()
#     if labels is None:
#         labels = two
#         inputs = one
#     else:
#         inputs = np.concatenate((inputs, one))
#         labels = np.concatenate((labels, two))
# all_scores, melodyEstimation, refMelody, inputs = calculate_metrics(dataobj, preds, labels, inputs, trackList, int(binsPerOctave), int(nOctave), testPath, batchSize, seqNumber, thresh=th, voicing=voicing)
#
# writeScores(all_scores, outPath)
# plot(outPath)

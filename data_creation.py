import medleydb as mdb
import numpy as np
import os, random, json, sys
from utils import log, binarize
import glob
import matplotlib
matplotlib.use('agg')
import librosa
import librosa.core as dsp
import librosa.display
import matplotlib.pyplot as plt
import sklearn
import re, csv
from scipy.signal import upfirdn
from scipy.ndimage import filters
from keras import backend as K
from model import model
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Input, Flatten, Lambda
from predict_on_audio import load_model

FMIN = 32.7
HOP_LENGTH = 256
Fs = 22050#44100#22050
HARMONICS = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
TASK = "MELODY2"

class DataSet(object):

    def __init__(self, inPath, outPath, targetDim):
        self.inputPath = inPath
        self.targetPath = outPath
        self.mtracks = mdb.load_all_multitracks(dataset_version = 'V1')
        self.trackList = mdb.TRACK_LIST_V1
        self.longest = 0
        if "rachel" in targetDim or "BASELINE" in targetDim:
            self.deepModel = load_model('melody2')
            self.deepModel.predict(np.zeros((2,360,50,6))) # Test/Initialize ?
            print(self.deepModel.summary())

    ### ---------------- CREATE DATASET OF FEATURES FROM AUDIO ---------------- ###
    def getFeature(self, dataSet, modelDim, outPath, binsPerOctave, nOctave, nHarmonics=1, homemade=False):

        log('Creating features from Audio')
        if os.path.isdir('/data2/anasynth_nonbp/laffitte'):
            audioPath = '/data2/anasynth_nonbp/laffitte/MedleyDB/Audio/'
            annotPath = '/data2/anasynth_nonbp/laffitte/MedleyDB/Annotations/Melody_Annotations/'+TASK
        else:
            annotPath = '/net/as-sdb/data/mir2/MedleyDB/Annotations/Melody_Annotations/'+TASK
            audioPath = '/net/as-sdb/data/mir2/MedleyDB/Audio/'
        self.trackList = []
        dirList = sorted(os.listdir(audioPath))
        dirList=[i for i in dirList if '.' not in i]
        H = []
        for j in dirList:  # loop over all audio files
            fileList = sorted(os.listdir(os.path.join(audioPath, j)))
            fileList = [k[:-8] for k in fileList if re.match('[^._].*?.wav', k)]
            tracks = [tr for tr in fileList if tr in dataSet]
            for i in tracks:
                prefix = i.replace('_MIX.wav', '')
                featPath = os.path.join(outPath, 'features')
                targPath = os.path.join(outPath, 'targets')
                if not os.path.isdir(featPath):
                    os.mkdir(featPath)
                if not os.path.isdir(targPath):
                    os.mkdir(targPath)
                inFile = os.path.join(featPath, '{}_mel2_input.npy'.format(prefix))
                outFile = os.path.join(targPath, '{}_mel2_target.npy'.format(prefix))
                if not os.path.exists(inFile) or not os.path.exists(outFile):
                    audioFile = os.path.join(audioPath, os.path.join(j, i+'_MIX.wav'))
                    annotFile = os.path.join(annotPath, i+'_'+TASK+'.csv')
                    if annotFile is not None and os.path.exists(annotFile):
                        ### Compute CQT
                        if not os.path.exists(inFile):
                            if homemade:
                                signal, fs = librosa.load(audioFile) # load signal
                                ''' Get CQT feature of whole signal according to the HCQT (Harmonic CQT) method described in "Deep Salience Representation for F0 Estimation in Polyphonic Music" by Rachel M. Bittner, Brian McFee, Justin Salamon, Peter Li, Juan P. Bello. The k-th CQT computes the CQT of the (k-1)-th harmonic of C1 '''
                                for k in range(nHarmonics): # loop over number of harmonics desired
                                    H.append(dsp.cqt(signal, Fs, fmin=(k+1)*FMIN, n_bins=binsPerOctave*nOctave, bins_per_octave=binsPerOctave, filter_scale=0.9, hop_length=HOP_LENGTH))
                            else: # use rachel's feature extraction
                                H = computeHcqt(audioFile, nHarmonics, binsPerOctave, nOctave)
                            np.save(inFile, H.astype(np.float32))
                        if not os.path.exists(outFile):
                            ### Get labels
                            data = readAnnotation(annotFile)
                            annot = np.asarray(np.array(data).T[0])
                            annot.reshape((np.shape(annot)[0], np.shape(annot)[1]))
                            times = annot[:,0]
                            freqs = annot[:,1]
                            freq_grid = librosa.cqt_frequencies(
                                binsPerOctave*nOctave, FMIN, bins_per_octave=binsPerOctave
                                )
                            length = int(np.floor(len(times)*(Fs/44100)))
                            time_grid = librosa.core.frames_to_time(
                                range(int(length)), sr=Fs, hop_length=HOP_LENGTH
                            )
                            target = createAnnotation(freq_grid, time_grid, times, freqs)
                            np.save(outFile, target.astype(np.float32))
                self.trackList.append(i.replace('_MIX.wav', '')) # Add current track to track list
                self.inputPath = featPath # Set feature path
                self.targetPath = targPath # Set target path

    def partDataset(self):

        # data_splits_path = '/net/inavouable/u.anasynth/laffitte/Code/ismir2017-deepsalience/outputs/data_splits.json'
        data_splits_path = '/u/anasynth/laffitte/datasplits.json'
        with open(data_splits_path, 'r') as fhandle:
            data_splits = json.load(fhandle)
        ### Get test set from Rachel's paper data
        testSet = data_splits['test']
        trainSet = data_splits['train']
        validSet = data_splits['validation']
        # restSet = [tr for tr in self.trackList if tr not in testSet]
        ### GET TRAIN AND VALID DATASET FROM THE REST OF MedleyDB DATASET
        # trainSet, validSet = sklearn.model_selection.train_test_split(restSet, train_size=0.70)
        return trainSet, validSet, testSet

    def sizeDataset(self, dataset, batchSize, rnnBatch=16):

        nTracks = 0 # count number of tracks
        nSamples = 0 # count number individual training examples
        nBlocks = 0 # number of batches
        bucketList = self.bucketDataset(dataset, rnnBatch)
        for (s, subTracks) in enumerate(bucketList):
            self.mtracks = mdb.load_all_multitracks(dataset_version = 'V1')
            tracks = [tr.track_id for tr in self.mtracks if tr.track_id in self.trackList]
            tracks = [tr for tr in tracks if tr in subTracks[0]]
            longest, _ = self.findLongest(tracks)
            nSequences = int(np.floor(longest/batchSize))
            nBlocks += nSequences
            for song in tracks:
                length, _ = self.findLongest([song])
                nSamples += length
            nTracks += len(tracks)
        return [nSamples, nBlocks, nTracks]

    def findLongest(self, dataset):

        longest = 0
        name = ''
        songDic = {}
        for track in dataset:
            i = glob.glob(os.path.join(self.inputPath, '{}_mel2_input.npy'.format(track)))
            j = glob.glob(os.path.join(self.targetPath, '{}_mel2_target.npy'.format(track)))
            if i and j:
                curTarget = np.load(j[0])
                if curTarget.shape[-1] >= longest:
                    longest = curTarget.shape[-1]
                    name = track
                songDic[track] = curTarget.shape[-1]
        return longest, name

    def bucketDataset(self, dataset, size):

        sortedList = []
        bucketList = []
        dataList = dataset[:]
        for k in range(len(dataset)):
            L, longest = self.findLongest(dataList)
            if L != 0:
                sortedList.append(longest)
                dataList.remove(longest)
        for k in range(int(np.floor(len(dataset)/size))+1):
            bucketList.append([])
            bucketList[-1].append(sortedList[k*size: k*size+size])

        return bucketList

    def formatDataset(self, myModel, dataset, timeDepth, targetDim, batchSize, hopSize, fftSize, nHarmonics, binsPerOctave, nOctave, voicing=False, rnnBatch=16, stateFull=True):

        self.mtracks = mdb.load_all_multitracks(dataset_version = 'V1')
        tracks = [tr.track_id for tr in self.mtracks if tr.track_id in self.trackList]
        tracks = [tr for tr in tracks if tr in dataset]
        # tracks = dataset
        while 1:
            bucketList = self.bucketDataset(tracks, rnnBatch)
            for (s, subTracks) in enumerate(bucketList):
                if stateFull:
                    log("Resetting model's states")
                    myModel.reset_states()
                if "SOFTMAX" in targetDim or "CATEGORICAL" in targetDim:
                    binary = True
                    voicing = True
                longest, _ = self.findLongest(subTracks[0])
                nSequences = int(np.floor(longest/batchSize))
                if voicing: # get size of longest track and set it as number of nBatches
                    nOuts = fftSize+1
                else:
                    nOuts = fftSize
                offset = 0
                for b in range(nSequences): # Iterate over total number of batches and fill them up 1-b-1
                    if "BASELINE" in targetDim or "MULTILABEL" in targetDim:
                        inputs = -1 * np.ones((rnnBatch, batchSize, fftSize))
                    else:
                        inputs = -1 * np.ones((rnnBatch, batchSize, fftSize, timeDepth, nHarmonics))
                    if "1D" in targetDim or "SOFTMAX" in targetDim or "BASELINE" in targetDim:
                        targets = -1 * np.ones((rnnBatch, batchSize, nOuts))
                    elif "2D" in targetDim:
                        targets = -1 * np.ones((rnnBatch, batchSize, nOuts, timeDepth))
                    elif "MULTILABEL" in targetDim: ### USING SEPARATE LABEL FOR PITCH AND OCTAVE DETECTION
                        targets = -1 * np.ones((rnnBatch, batchSize, nOuts))
                        targetNote = -1 * np.ones((rnnBatch, batchSize, binsPerOctave, timeDepth))
                        targetOctave = -1 * np.ones((rnnBatch, batchSize, nOctave, timeDepth))
                    for (k, track) in enumerate(sorted(subTracks[0])):
                        i = glob.glob(os.path.join(self.inputPath, '{}_mel2_input.npy'.format(track)))
                        j = glob.glob(os.path.join(self.targetPath, '{}_mel2_target.npy'.format(track)))
                        if i and j:
                            curInput = np.load(i[0])
                            # idx = np.array(np.arange(0,curInput.shape[-1],44100/Fs), dtype='int32')
                            # curInput = curInput[:,:,idx]
                            curTarget = np.load(j[0])
                            if curInput.shape[-1] != curTarget.shape[-1]:
                                lim = np.min((curInput.shape[-1], curTarget.shape[-1]))
                                curInput = curInput[:,:,:lim]
                                curTarget = curTarget[:,:lim]
                            # curInput = zero_pad(curInput, True, timeDepth)
                            if "1D" in targetDim or "CATEGORICAL" in targetDim:
                                if offset+batchSize+timeDepth < curInput.shape[-1]:
                                    for kk in range(0, batchSize, hopSize):
                                        if len(curInput.shape) == 2:
                                            temp = curInput[None, :, offset+kk:offset+kk+timeDepth]
                                        else:
                                            temp = curInput[:,:,offset+kk:offset+kk+timeDepth]
                                        inputs[k,kk] = temp.transpose(1,2,0)
                                    tar = curTarget[:,offset:offset+batchSize].transpose(1,0)
                                if voicing:
                                    if len(tar.shape)<2:
                                        targets[k,:] = computeVoicing(tar)
                                    else:
                                        targets[k,:] = -1 * np.ones((batchSize, tar.shape[1]+1))
                                        for m in range(tar.shape[0]):
                                            targets[k,m] = computeVoicing(tar[m, :])
                                else:
                                    targets[k,:] = tar
                            elif "BASELINE" in targetDim or "2D" in targetDim:
                                if offset+batchSize < curTarget.shape[-1]:
                                    temp = curInput[:,:,offset:offset+batchSize]
                                    temp = temp.transpose(1, 2, 0)[None,:,:,:]
                                    if "BASELINE" in targetDim:
                                        temp = self.deepModel.predict(temp)
                                    inputs[k,:,:] = temp[0,:,:].transpose(1,0)
                                    tar = curTarget[:,offset:offset+batchSize].transpose(1,0)
                                    if voicing:
                                        if len(tar.shape)<2:
                                            targets[k,:,:] = computeVoicing(tar)
                                        else:
                                            targets[k,:,:] = -1 * np.ones((batchSize, tar.shape[1]+1))
                                            for m in range(tar.shape[0]):
                                                targets[k,m] = computeVoicing(tar[m, :])
                                    else:
                                        targets[k,:,:] = tar
                    offset += batchSize
                    if "MULTILABEL" in targetDim:
                        targetNote, targetOctave = splitTarget(targets, nOctave, binsPerOctave)
                        yield inputs, {'note':targetNote, 'octave':targetOctave}
                    else:
                        yield inputs, targets

    def toyData(self, myModel, dataset, rnnBatch, batchSize, targetDim, fftSize, stateFull):
        while 1:
            bucketList = self.bucketDataset(dataset, rnnBatch)
            for (s, subTracks) in enumerate(bucketList):
                if stateFull:
                    log("Resetting model's states")
                    myModel.reset_states()
                if "SOFTMAX" in targetDim or "CATEGORICAL" in targetDim:
                    binary = True
                    voicing = True
                self.mtracks = mdb.load_all_multitracks(dataset_version = 'V1')
                tracks = [tr.track_id for tr in self.mtracks if tr.track_id in self.trackList]
                tracks = [tr for tr in tracks if tr in subTracks[0]]
                longest, _ = self.findLongest(tracks)
                nSequences = int(np.floor(longest/batchSize))
                offset = 0
                for b in range(nSequences):
                    labels = -1 * np.ones((rnnBatch, batchSize, fftSize))
                    inputs = -1 * np.ones((rnnBatch, batchSize, fftSize))
                    for l in range(2, rnnBatch):
                        inputs[l,:,:] = l * np.ones((batchSize, fftSize))
                        labels[l,:,:] = l * np.ones((batchSize, fftSize))

                    yield inputs, labels

def splitTarget(labels, nOctave, binsPerOctave):

    shape = labels.shape
    note = np.zeros((shape[0], shape[1], binsPerOctave))
    octave = np.zeros((shape[0], shape[1], nOctave))
    for b in range(shape[0]):
        for f in range(shape[1]):
            if labels[b,f,:].nonzero()[0].any():
                curOctave = np.floor(labels[b,f,:].nonzero()[0]/(binsPerOctave))
                curNote = np.floor(labels[b,f,:].nonzero()[0]%binsPerOctave)
                octave[b,f,curOctave] = 1
                note[b,f,curNote] = 1

    return note, octave

def mergeTarget(note, octave):
    shape = note.shape
    binsPerOctave = shape[2]
    nOctave = octave.shape[2]
    target = np.zeros((shape[0], shape[1], binsPerOctave*nOctave))
    for b in shape[0]:
        for f in shape[1]:
            if note[b,f,:].nonzero()[0].any():
                curNote = note[b,f,:].nonzero()[0]
                curOctave = octave[b,f,:].nonzero()[0]
                target[b,f,:] = octave*note

def getLabels(dataobj, dataset, params, modelDim, voicing, fftSize, rnnBatch):
    log('Getting labels and inputs for plotting and score calculation')
    batchSize = int(params['batchSize'])
    timeDepth = int(params['timeDepth'])
    nHarmonics = int(params['nHarmonics'])
    hopSize = int(params['hopSize'])
    stateFull = True if params['stateFull']=="True" else False
    trackList = []
    bucketList = dataobj.bucketDataset(dataset, rnnBatch)
    labels = [None] * len(dataset)
    inputs = [None] * len(dataset)
    for (s, subTracks) in enumerate(bucketList):
        for (k, track) in enumerate(sorted(subTracks[0])):
            trackList.append(track)
    for (s, subTracks) in enumerate(bucketList):
        longest, _ = dataobj.findLongest(subTracks[0])
        nSequences = int(np.floor(longest/batchSize))
        if voicing: # get size of longest track and set it as number of nBatches
            nOuts = fftSize+1
        else:
            nOuts = fftSize
        offset = 0
        for b in range(nSequences): # Iterate over total number of batches and fill them up 1-b-1
            if "BASELINE" in modelDim:
                inp = -1 * np.ones((rnnBatch, batchSize, fftSize))
            else:
                inp = -1 * np.ones((rnnBatch, batchSize, fftSize, timeDepth, nHarmonics))
            if "1D" in modelDim or "SOFTMAX" in modelDim or "BASELINE" in modelDim:
                targets = -1 * np.ones((rnnBatch, batchSize, nOuts))
            elif "2D" in modelDim:
                targets = -1 * np.ones((rnnBatch, batchSize, nOuts, timeDepth))
            elif "MULTILABEL" in modelDim: ### USING SEPARATE LABEL FOR PITCH AND OCTAVE DETECTION
                targets = -1 * np.ones((rnnBatch, batchSize, nOuts))
                targetNote = -1 * np.ones((rnnBatch, batchSize, binsPerOctave, timeDepth))
                targetOctave = -1 * np.ones((rnnBatch, batchSize, nOctave, timeDepth))
            for (k, track) in enumerate(sorted(subTracks[0])):
                i = glob.glob(os.path.join(dataobj.inputPath, '{}_mel2_input.npy'.format(track)))
                j = glob.glob(os.path.join(dataobj.targetPath, '{}_mel2_target.npy'.format(track)))
                curInput = np.load(i[0])
                curTarget = np.load(j[0])
                if curInput.shape[-1] != curTarget.shape[-1]:
                    lim = np.min((curInput.shape[-1], curTarget.shape[-1]))
                    curInput = curInput[:,:,:lim]
                    curTarget = curTarget[:,:lim]
                if i and j:
                    if offset+batchSize < curTarget.shape[-1]:
                        temp = curInput[:,:,offset:offset+batchSize]
                        temp = temp.transpose(1, 2, 0)[None,:,:,:]
                        if "BASELINE" in modelDim:
                            temp = dataobj.deepModel.predict(temp)
                        inp[k,:,:] = temp[0,:,:].transpose(1,0)
                        tar = curTarget[:,offset:offset+batchSize].transpose(1,0)
                        if voicing:
                            if len(tar.shape)<2:
                                targets[k,:,:] = computeVoicing(tar)
                            else:
                                targets[k,:,:] = -1 * np.ones((batchSize, tar.shape[1]+1))
                                for m in range(tar.shape[0]):
                                    targets[k,m] = computeVoicing(tar[m, :])
                        else:
                            targets[k,:,:] = tar
                        if labels[s*rnnBatch+k] is None:
                            labels[s*rnnBatch+k] = targets[k,:,:]
                            inputs[s*rnnBatch+k] = inp[k,:,:]
                        else:
                            labels[s*rnnBatch+k] = np.concatenate((labels[s*rnnBatch+k], targets[k,:,:]), 0)
                            inputs[s*rnnBatch+k] = np.concatenate((inputs[s*rnnBatch+k], inp[k,:,:]), 0)
            offset += batchSize
    return labels, inputs, trackList

def getLabelMatrix(myModel, dataobj, dataset, params, modelDim, voicing, fftSize, rnnBatch):
    batchSize = int(params['batchSize'])
    timeDepth = int(params['timeDepth'])
    nHarmonics = int(params['nHarmonics'])
    hopSize = int(params['hopSize'])
    binsPerOctave = int(params['binsPerOctave'])
    nOctave = int(params['nOctave'])
    stateFull = True if params['stateFull']=="True" else False
    gen = dataobj.formatDataset(myModel, dataset, int(timeDepth), modelDim, batchSize, hopSize, fftSize, nHarmonics, binsPerOctave, nOctave, voicing, rnnBatch, stateFull)
    nSamples, size, length = dataobj.sizeDataset(dataset, batchSize, rnnBatch)
    trackList = []
    bucketList = dataobj.bucketDataset(dataset, rnnBatch)
    for (s, subTracks) in enumerate(bucketList):
        print(s)
        for (k, track) in enumerate(sorted(subTracks[0])):
            print(track)
            trackList.append(track)
    if nSamples != 0:
        labels = None
        inputs = None
        print("Size:", size)
        for l in range(size):
            one, two = gen.__next__()
            print(l)
            if labels is None:
                labels = two
                inputs = one
            else:
                inputs = np.concatenate((inputs, one))
                labels = np.concatenate((labels, two))
    return labels, inputs, trackList

def createAnnotation(freq_grid, time_grid, annotation_times,
                             annotation_freqs):
    """Create the binary annotation target labels
    """

    n_freqs = len(freq_grid)
    n_times = len(time_grid)
    idx = np.array(np.arange(0, len(annotation_times), 44100/Fs), dtype='int32')
    annotation_times = annotation_times[idx]
    annotation_freqs = annotation_freqs[idx]

    # time_bins = grid_to_bins(time_grid, 0.0, time_grid[-1])
    time_bins = time_grid
    freq_bins = grid_to_bins(freq_grid, 0.0, freq_grid[-1])
    annot_time_idx = np.digitize(annotation_times, time_bins) - 1
    annot_freq_idx = np.digitize(annotation_freqs, freq_bins) - 1

    annotation_target = np.zeros((n_freqs+1, n_times))
    annotation_target[annot_freq_idx, annot_time_idx] = 1

    return annotation_target[1:,:]

def computeVoicing(activations):

    # import pdb; pdb.set_trace()
    if len(activations.shape)==1:
        ind = np.where((activations==0),0,0)
        voicingArray = activations
        voicingArray = np.insert(voicingArray, -1, 0)
        if not any(activations):
            voicingArray[ind] = 1
    else:
        for k in range(activations.shape[0]):
            ind = np.where((activations[k,:]==0),0,0)
            voicingArray = activations
            voicingArray = np.insert(voicingArray, -1, 0)
            if not any(activations[k,:]):
                voicingArray[ind] = 1

    return voicingArray

### ---------------- COMPUTE HCQT FEATURES ---------------- ###
def computeHcqt(audio_fpath, nHarmonics, binsPerOctave, nOctave):

    y, fs = librosa.load(audio_fpath, sr=Fs)
    cqt_list = []
    shapes = []
    harmonics = HARMONICS[0:nHarmonics]
    for h in harmonics:
        cqt = librosa.cqt(
            y, sr=Fs, hop_length=HOP_LENGTH, fmin=FMIN*float(h),
            n_bins=binsPerOctave*nOctave,
            bins_per_octave=binsPerOctave
        )
        cqt_list.append(cqt)
        shapes.append(cqt.shape)

    shapes_equal = [s == shapes[0] for s in shapes]
    if not all(shapes_equal):
        min_time = np.min([s[1] for s in shapes])
        new_cqt_list = []
        for i in range(len(cqt_list)):
            new_cqt_list.append(cqt_list[i][:, :min_time])
        cqt_list = new_cqt_list

    log_hcqt = ((1.0/80.0) * librosa.core.amplitude_to_db(
        np.abs(np.array(cqt_list)), ref=np.max)) + 1.0

    return log_hcqt

def turnToVoicing(targets):
    ### ---------------- TURN ANNOTATION TO BINARY VOICING LABELS ---------------- ###
    return

def readAnnotation(filePath, num_cols=None, header=False):
    if filePath is not None and os.path.exists(filePath):
        with open(filePath) as f_handle:
            annotation = []
            linereader = csv.reader(f_handle)

            # skip the headers for non csv files
            if header:
                header = next(linereader)
            else:
                header = []

            for line in linereader:
                if num_cols:
                    line = line[:num_cols]
                annotation.append([float(val) for val in line])
        return annotation, header
    else:
        return None, None

def grid_to_bins(grid, start_bin_val, end_bin_val):
    """Compute the bin numbers from a given grid
    """
    bin_centers = (grid[1:] + grid[:-1])/2.0
    bins = np.concatenate([[start_bin_val], bin_centers, [end_bin_val]])
    return bins

def bkld(y_true, y_pred):
    """KL Divergence where both y_true an y_pred are probabilities
    """
    y_true = K.clip(y_true, K.epsilon(), 1.0 - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return K.mean(K.mean(
        -1.0*y_true* K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred),
        axis=-1), axis=-1)

def zero_pad(curInput, sym, size):

    shape = np.shape(curInput)
    newshape = np.asarray(shape)
    if len(shape)>=3:
        newshape[2] = newshape[2] + size
        newStructure = np.zeros(newshape)
        start = int(size / 2)
        stop = int(newshape[2] - (size / 2))
        newStructure[:, :, start : stop] = curInput
    else:
        newshape[1] = newshape[1] + size
        newStructure = np.zeros(newshape)
        start = int(size / 2)
        stop = int(newshape[1] - (size / 2))
        newStructure[:, start : stop] = curInput

    return newStructure

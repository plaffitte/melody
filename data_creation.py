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
Fs = 22050
HARMONICS = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

class DataSet(object):

    def __init__(self, inPath, outPath, targetDim):
        self.inputPath = inPath
        self.targetPath = outPath
        self.mtracks = mdb.load_all_multitracks(dataset_version = 'V1')
        self.trackList = mdb.TRACK_LIST_V1
        self.longest = 0
        if "rachel" in targetDim:
            self.deepModel = load_model('melody2')
            toto = self.deepModel.predict(np.zeros((2,360,50,6))) # Test/Initialize ?
            print(self.deepModel.summary())

    ### ---------------- CREATE DATASET OF FEATURES FROM AUDIO ---------------- ###
    def getFeature(self, dataSet, modelDim, outPath, binsPerOctave, nOctave, nHarmonics=1, homemade=False):

        log('Creating features from Audio')
        if os.path.isdir('/data2/anasynth_nonbp/laffitte'):
            audioPath = '/data2/anasynth_nonbp/laffitte/MedleyDB/Audio/'
            annotPath = '/data2/anasynth_nonbp/laffitte/MedleyDB/Annotations/Melody_Annotations/MELODY1/'
        else:
            annotPath = '/net/as-sdb/data/mir2/MedleyDB/Annotations/Melody_Annotations/MELODY1/'
            audioPath = '/net/as-sdb/data/mir2/MedleyDB/Audio/'
        #
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
                inFile = os.path.join(featPath, '{}_mel1_input.npy'.format(prefix))
                outFile = os.path.join(targPath, '{}_mel1_output.npy'.format(prefix))
                if not os.path.exists(inFile):
                    audioFile = os.path.join(audioPath, os.path.join(j, i+'_MIX.wav'))
                    annotFile = os.path.join(annotPath, i+'_MELODY1.csv')
                    if annotFile is not None and os.path.exists(annotFile):
                        ### Compute CQT
                        if homemade:
                            signal, fs = librosa.load(audioFile) # load signal
                            ''' Get CQT feature of whole signal according to the HCQT (Harmonic CQT) method described in "Deep Salience Representation for F0 Estimation in Polyphonic Music" by Rachel M. Bittner, Brian McFee, Justin Salamon, Peter Li, Juan P. Bello. The k-th CQT computes the CQT of the (k-1)-th harmonic of C1 '''
                            for k in range(nHarmonics): # loop over number of harmonics desired
                                H.append(dsp.cqt(signal, Fs, fmin=(k+1)*FMIN, n_bins=binsPerOctave*nOctave, bins_per_octave=binsPerOctave, filter_scale=0.9, hop_length=HOP_LENGTH))
                        else: # use rachel's feature extraction
                            H = computeHcqt(audioFile, nHarmonics, binsPerOctave, nOctave)
                        ### Get labels
                        data = readAnnotation(annotFile)
                        annot = np.asarray(np.array(data).T[0])
                        annot.reshape((np.shape(annot)[0], np.shape(annot)[1]))
                        times = annot[:,0]
                        freqs = annot[:,1]
                        idx = np.where(freqs != 0.0)[0]
                        times = times[idx]
                        freqs = freqs[idx]
                        # X, Y, f, t = get_input_output_pairs(H, times, freqs, True)

                        freq_grid = librosa.cqt_frequencies(
                            binsPerOctave*nOctave, FMIN, bins_per_octave=binsPerOctave
                            )
                        time_grid = librosa.core.frames_to_time(
                            range(len(H[0][0])), sr=Fs, hop_length=HOP_LENGTH
                            )
                        target = createAnnotation(time_grid, freq_grid, times, freqs)

                        if not os.path.isdir(featPath):
                            os.mkdir(featPath)
                        if not os.path.isdir(targPath):
                            os.mkdir(targPath)

                        np.save(inFile, H.astype(np.float32))
                        np.save(outFile, target.astype(np.float32))
                self.trackList.append(i.replace('_MIX.wav', '')) # Add current track to track list
                self.inputPath = featPath # Set feature path
                self.targetPath = targPath # Set target path

    def partDataset(self):

        data_splits_path = '/net/inavouable/u.anasynth/laffitte/Code/ismir2017-deepsalience/outputs/data_splits.json'
        with open(data_splits_path, 'r') as fhandle:
            data_splits = json.load(fhandle)
        ### Get test set from Rachel's paper data
        testSet = data_splits['test']
        restSet = [tr for tr in self.trackList if tr not in testSet]
        ### GET TRAIN AND VALID DATASET FROM THE REST OF MedleyDB DATASET
        trainSet, validSet = sklearn.model_selection.train_test_split(restSet, train_size=0.80)
        return trainSet, validSet, testSet

    def sizeDataset(self, dataset, timeDepth, batchSize, hopSize, fftSize, nHarmonics, targetDim, rnnBatch=16, stateFull=True):
        length = 0 # count number of tracks
        nSamples = 0 # count number individual training examples
        nBlocks = 0 # number of batches
        if stateFull:
            longest, _ = self.findLongest(dataset)
            for track in dataset[:]:
                i = glob.glob(os.path.join(self.inputPath, '{}_mel1_input.npy'.format(track)))
                if i:
                    curInput = np.load(i[0])
                    # if curInput.shape[-1] >= (longest - 100):
                    #     dataset.remove(track)
                    #     print "removing long track:", track
                    # else:
                    nSamples += curInput.shape[-1]
            longest, _ = self.findLongest(dataset)
            nBlocks = int(np.floor(longest / batchSize))
            # nSamples = nBlocks*batchSize*fftSize*timeDepth*nHarmonics
            length = len(dataset)
        else:
            for track in dataset[:]:
                i = glob.glob(os.path.join(self.inputPath, '{}_mel1_input.npy'.format(track)))
                j = glob.glob(os.path.join(self.targetPath, '{}_mel1_output.npy'.format(track)))
                if i and j:
                    curInput = np.load(i[0])
                    n_t = curInput.shape[-1]
                    L = batchSize * hopSize
                    nBatches = int(np.round((n_t - timeDepth) / L) + 1)
                    for l in range(nBatches):
                        stride = np.arange(0, L, hopSize)
                        for k in stride:
                            if l * L + k <= (n_t - timeDepth):
                                nSamples += timeDepth
                        nBlocks += 1
                    length += 1
                else:
                    sys.stdout.flush()
                    log("NO FILE FOUND FOR SONG:\n", track)
        return [nSamples, nBlocks, length]

    def findLongest(self, dataset):
        longest = 0
        name = ''
        songDic = {}
        for track in dataset:
            i = glob.glob(os.path.join(self.inputPath, '{}_mel1_input.npy'.format(track)))
            j = glob.glob(os.path.join(self.targetPath, '{}_mel1_output.npy'.format(track)))
            if i and j:
                curTarget = np.load(j[0])
                if curTarget.shape[-1] >= longest:
                    longest = curTarget.shape[-1]
                    name = track
                songDic[track] = curTarget.shape[-1]
        # sortedDic = sorted(songDic, key=songDic.get)
        # for song in sortedDic:
        #     print song, songDic[song]
        return longest, name

    ### ---------------- RETURN A GENERATOR FO60R DATA AND LABELS ---------------- ###
    def formatDataset(self, dataset, timeDepth, targetDim, batchSize, hopSize, fftSize, nHarmonics, voicing=False, binary=False):
        while 1:
            if "SOFTMAX" in targetDim or "VOICING" in targetDim:
                binary = True
                voicing = True
            self.mtracks = mdb.load_all_multitracks(dataset_version = 'V1')
            tracks = [tr.track_id for tr in self.mtracks if tr.track_id in self.trackList]
            tracks = [tr for tr in tracks if tr in dataset]
            for track in tracks:
                i = glob.glob(os.path.join(self.inputPath, '{}_mel1_input.npy'.format(track)))
                j = glob.glob(os.path.join(self.targetPath, '{}_mel1_output.npy'.format(track)))
                if i and j:
                    curInput = np.load(i[0])
                    curTarget = np.load(j[0])
                    if targetDim=="1D" or "SOFTMAX" in targetDim:
                        curInput = zero_pad(curInput, True, timeDepth)
                    L = batchSize * hopSize
                    nRounds = int(np.round((curTarget.shape[-1] - timeDepth) / L) + 1)
                    for l in range(nRounds):
                        if "2D" in targetDim or "rachel" in targetDim:
                            inputs = np.zeros((batchSize, fftSize, timeDepth, nHarmonics))
                        else:
                            inputs = np.zeros((batchSize, fftSize, timeDepth, nHarmonics))
                        if "VOICING" in targetDim:
                            nOuts = fftSize+1
                        else:
                            nOuts = fftSize
                        if "1D" in targetDim:
                            targets = np.zeros((batchSize, nOuts))
                        elif "2D" in targetDim or "rachel" in targetDim or "BASELINE" in targetDim:
                            targets = np.zeros((batchSize, timeDepth, nOuts))
                        stride = np.arange(0, L, hopSize)
                        indBatch = 0
                        for k in stride:
                            if l * L + k <= (curTarget.shape[-1] - timeDepth):
                                voicingVector = None
                                if "2D" in targetDim or "rachel" in targetDim:
                                    inputs[indBatch,:,:,:] = curInput[:,:,l*L+k:l*L+k+timeDepth].transpose(1,2,0)
                                else:
                                    inputs[indBatch,:,:,:] = curInput[:,:,l*L+k:l*L+k+timeDepth].transpose(1,2,0)
                                if "1D" in targetDim:
                                    tar = curTarget[:, l * L + k]
                                else:
                                    tar = curTarget[:, l * L + k : l * L + k + timeDepth].T
                                if binary:
                                    tar = binarize(tar)
                                if voicing:
                                    if len(targets.shape)<=2:
                                        targets[indBatch, :] = computeVoicing(tar)
                                    else:
                                        for m in range(tar.shape[0]):
                                            targets[indBatch, m, :] = computeVoicing(tar[m, :])
                                else:
                                    targets[indBatch, :] = tar
                                indBatch += 1
                        if "rachel" in targetDim or "BASELINE" in targetDim:
                            inputs = self.deepModel.predict(inputs)
                            # targets = targets[None,:,:]
                        if "1D" in targetDim and not "rachel" in targetDim:
                            yield [inputs[None,:,:,:,:], targets[None,:,:]]
                        else:
                            yield [inputs, targets]
                else:
                    pass
                    # log("No path found for this file")

    def formatDatasetStatefull(self, dataset, timeDepth, targetDim, batchSize, hopSize, fftSize, nHarmonics, voicing=False, binary=False, rnnBatch=16):
        _, nSequences, _ = self.sizeDataset(dataset, timeDepth, batchSize, hopSize, fftSize, nHarmonics, targetDim, rnnBatch, True)
#########################################################################################################
##########################################################################################################
##########################################################################################################
########--------->>>>> TRYING TO USE FIT GENERATOR METHOD <<<<<<--------------------------------------
        bucketList = bucketDataset(dataset, rnnBatch, dataobj)
        for (s, subTracks) in enumerate(bucketList):
            log("Training on subset {}".format(s))
            myModel.reset_states()
            _, subSteps, nSongsSubset = dataobj.sizeDataset(subTracks[0], timeDepth, batchSize, hopSize, fftSize, nHarmonics, modelDim, rnnBatch, stateFull)
            if "SOFTMAX" in targetDim or "CATEGORICAL" in targetDim:
                binary = True
                voicing = True
            self.mtracks = mdb.load_all_multitracks(dataset_version = 'V1')
            tracks = [tr.track_id for tr in self.mtracks if tr.track_id in self.trackList]
            tracks = [tr for tr in tracks if tr in dataset]
            # get size of longest track and set it as number of nBatches
            if voicing:
                nOuts = fftSize+1
            else:
                nOuts = fftSize
            offset = 0
            for b in range(nSequences): # Iterate over total number of batches and fill them up 1-b-1
                if "BASELINE" in targetDim:
                    # inputs = np.zeros((rnnBatch, fftSize, batchSize, nHarmonics))
                    inputs = np.zeros((rnnBatch, batchSize, fftSize))
                else:
                    inputs = np.zeros((rnnBatch, batchSize, fftSize, timeDepth, nHarmonics))
                if "1D" in targetDim or "SOFTMAX" in targetDim or "BASELINE" in targetDim:
                    targets = np.zeros((rnnBatch, batchSize, nOuts))
                elif "2D" in targetDim:
                    targets = np.zeros((rnnBatch, batchSize, nOuts, timeDepth))
                elif "MULTILABEL" in targetDim:
                    targetNote = np.zeros((rnnBatch, batchSize, binsPerOctave, timeDepth))
                    targetOctave = np.zeros((rnnBatch, batchSize, nOctave, timeDepth))
                for (k, track) in enumerate(tracks):
                    i = glob.glob(os.path.join(self.inputPath, '{}_mel1_input.npy'.format(track)))
                    j = glob.glob(os.path.join(self.targetPath, '{}_mel1_output.npy'.format(track)))
                    if i and j:
                        curInput = np.load(i[0])
                        curTarget = np.load(j[0])
                        curInput = zero_pad(curInput, True, timeDepth)
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
                                    targets[k,:] = np.zeros((batchSize, tar.shape[1]+1))
                                    for m in range(tar.shape[0]):
                                        targets[k,m] = computeVoicing(tar[m, :])
                            else:
                                targets[k,:] = tar
                        elif "BASELINE" in targetDim or "2D" in targetDim:
                            if offset+batchSize < curTarget.shape[-1]:
                                temp = curInput[:,offset:offset+batchSize]
                                inputs[k,:,:] = temp.transpose(1,0)
                                tar = curTarget[:,offset:offset+batchSize].transpose(1,0)
                                if voicing:
                                    if len(tar.shape)<2:
                                        targets[k,:] = computeVoicing(tar)
                                    else:
                                        targets[k,:] = np.zeros((batchSize, tar.shape[1]+1))
                                        for m in range(tar.shape[0]):
                                            targets[k,m] = computeVoicing(tar[m, :])
                                else:
                                    targets[k,:] = tar
                offset += batchSize
                # if "BASELINE" in targetDim:
                #     inputs = self.deepModel.predict(inputs).transpose(0,2,1)
                if "MULTILABEL" in targetDim:
                    targetNote, targetOctave = splitTarget(targets)
                    yield inputs, [targetNote, targetOctave], False
                else:
                    yield inputs, targets, False

##########################################################################################################
##########################################################################################################
##########################################################################################################

        while 1:
            # if "SOFTMAX" in targetDim or "CATEGORICAL" in targetDim:
            #     binary = True
            #     voicing = True
            # self.mtracks = mdb.load_all_multitracks(dataset_version = 'V1')
            # tracks = [tr.track_id for tr in self.mtracks if tr.track_id in self.trackList]
            # tracks = [tr for tr in tracks if tr in dataset]
            # # get size of longest track and set it as number of nBatches
            # if voicing:
            #     nOuts = fftSize+1
            # else:
            #     nOuts = fftSize
            # offset = 0
            # for b in range(nSequences): # Iterate over total number of batches and fill them up 1-b-1
            #     if "BASELINE" in targetDim:
            #         # inputs = np.zeros((rnnBatch, fftSize, batchSize, nHarmonics))
            #         inputs = np.zeros((rnnBatch, batchSize, fftSize))
            #     else:
            #         inputs = np.zeros((rnnBatch, batchSize, fftSize, timeDepth, nHarmonics))
            #     if "1D" in targetDim or "SOFTMAX" in targetDim or "BASELINE" in targetDim:
            #         targets = np.zeros((rnnBatch, batchSize, nOuts))
            #     elif "2D" in targetDim:
            #         targets = np.zeros((rnnBatch, batchSize, nOuts, timeDepth))
            #     elif "MULTILABEL" in targetDim:
            #         targetNote = np.zeros((rnnBatch, batchSize, binsPerOctave, timeDepth))
            #         targetOctave = np.zeros((rnnBatch, batchSize, nOctave, timeDepth))
            #     for (k, track) in enumerate(tracks):
            #         i = glob.glob(os.path.join(self.inputPath, '{}_mel1_input.npy'.format(track)))
            #         j = glob.glob(os.path.join(self.targetPath, '{}_mel1_output.npy'.format(track)))
            #         if i and j:
            #             curInput = np.load(i[0])
            #             curTarget = np.load(j[0])
            #             curInput = zero_pad(curInput, True, timeDepth)
            #             if "1D" in targetDim or "CATEGORICAL" in targetDim:
            #                 if offset+batchSize+timeDepth < curInput.shape[-1]:
            #                     for kk in range(0, batchSize, hopSize):
            #                         if len(curInput.shape) == 2:
            #                             temp = curInput[None, :, offset+kk:offset+kk+timeDepth]
            #                         else:
            #                             temp = curInput[:,:,offset+kk:offset+kk+timeDepth]
            #                         inputs[k,kk] = temp.transpose(1,2,0)
            #                     tar = curTarget[:,offset:offset+batchSize].transpose(1,0)
            #                 if voicing:
            #                     if len(tar.shape)<2:
            #                         targets[k,:] = computeVoicing(tar)
            #                     else:
            #                         targets[k,:] = np.zeros((batchSize, tar.shape[1]+1))
            #                         for m in range(tar.shape[0]):
            #                             targets[k,m] = computeVoicing(tar[m, :])
            #                 else:
            #                     targets[k,:] = tar
            #             elif "BASELINE" in targetDim or "2D" in targetDim:
            #                 if offset+batchSize < curTarget.shape[-1]:
            #                     temp = curInput[:,offset:offset+batchSize]
            #                     inputs[k,:,:] = temp.transpose(1,0)
            #                     tar = curTarget[:,offset:offset+batchSize].transpose(1,0)
            #                     if voicing:
            #                         if len(tar.shape)<2:
            #                             targets[k,:] = computeVoicing(tar)
            #                         else:
            #                             targets[k,:] = np.zeros((batchSize, tar.shape[1]+1))
            #                             for m in range(tar.shape[0]):
            #                                 targets[k,m] = computeVoicing(tar[m, :])
            #                     else:
            #                         targets[k,:] = tar
            #     offset += batchSize
            #     # if "BASELINE" in targetDim:
            #     #     inputs = self.deepModel.predict(inputs).transpose(0,2,1)
            #     if "MULTILABEL" in targetDim:
            #         targetNote, targetOctave = splitTarget(targets)
            #         yield inputs, [targetNote, targetOctave], False
            #     else:
            #         yield inputs, targets, False

def createAnnotation(time_grid, freq_grid, time, freq):

    time_bins = grid_to_bins(time_grid, 0.0, time_grid[-1])
    freq_bins = grid_to_bins(freq_grid, 0.0, freq_grid[-1])

    annot_time_idx = np.digitize(time, time_bins) - 1
    annot_freq_idx = np.digitize(freq, freq_bins) - 1

    n_freqs = len(freq_grid)
    n_times = len(time_grid)

    idx = annot_time_idx < n_times
    annot_time_idx = annot_time_idx[idx]
    annot_freq_idx = annot_freq_idx[idx]

    idx2 = annot_freq_idx < n_freqs
    annot_time_idx = annot_time_idx[idx2]
    annot_freq_idx = annot_freq_idx[idx2]

    annotation_target = np.zeros((n_freqs, n_times))
    annotation_target[annot_freq_idx, annot_time_idx] = 1

    return annotation_target

def computeVoicing(activations):

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

### ---------------- TURN ANNOTATION TO BINARY VOICING LABELS ---------------- ###
def turnToVoicing(targets):
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

def get_input_output_pairs(hcqt, annot_times, annot_freqs, gaussian_blur):

    freq_grid = librosa.cqt_frequencies(BINS_PER_OCTAVE*N_OCTAVES, FMIN, bins_per_octave=BINS_PER_OCTAVE)
    time_grid = librosa.core.frames_to_time(range(len(hcqt[0][0])), sr=Fs, hop_length=HOP_LENGTH)
    annot_target = create_annotation_target(freq_grid, time_grid, annot_times, annot_freqs, gaussian_blur)

    return hcqt, annot_target, freq_grid, time_grid

def grid_to_bins(grid, start_bin_val, end_bin_val):
    """Compute the bin numbers from a given grid
    """
    bin_centers = (grid[1:] + grid[:-1])/2.0
    bins = np.concatenate([[start_bin_val], bin_centers, [end_bin_val]])
    return bins

def create_annotation_target(freq_grid, time_grid, annotation_times,annotation_freqs, gaussian_blur):
    """Create the binary annotation target labels
    """
    time_bins = grid_to_bins(time_grid, 0.0, time_grid[-1])
    freq_bins = grid_to_bins(freq_grid, 0.0, freq_grid[-1])

    annot_time_idx = np.digitize(annotation_times, time_bins) - 1
    annot_freq_idx = np.digitize(annotation_freqs, freq_bins) - 1

    n_freqs = len(freq_grid)
    n_times = len(time_grid)

    idx = annot_time_idx < n_times
    annot_time_idx = annot_time_idx[idx]
    annot_freq_idx = annot_freq_idx[idx]

    idx2 = annot_freq_idx < n_freqs
    annot_time_idx = annot_time_idx[idx2]
    annot_freq_idx = annot_freq_idx[idx2]

    annotation_target = np.zeros((n_freqs, n_times))
    annotation_target[annot_freq_idx, annot_time_idx] = 1

    if gaussian_blur:
        annotation_target_blur = filters.gaussian_filter1d(
            annotation_target, 1, axis=0, mode='constant'
        )
        if len(annot_freq_idx) > 0:
            min_target = np.min(
                annotation_target_blur[annot_freq_idx, annot_time_idx]
            )
        else:
            min_target = 1.0

        annotation_target_blur = annotation_target_blur / min_target
        annotation_target_blur[annotation_target_blur > 1.0] = 1.0

    return annotation_target_blur

def bkld(y_true, y_pred):
    """KL Divergence where both y_true an y_pred are probabilities
    """
    y_true = K.clip(y_true, K.epsilon(), 1.0 - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return K.mean(K.mean(
        -1.0*y_true* K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred),
        axis=-1), axis=-1)

def toyData(shape):
    data = np.random.random(shape)
    labels = np.random.uniform(0, 1, size=shape)

    return data, labels

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

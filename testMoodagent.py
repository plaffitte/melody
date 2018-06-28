import numpy as np
import os, glob
import librosa
import librosa.core as dsp
import librosa.display
import sys
from predict_on_audio import load_model
# from keras.models import load_model
from data_creation import toyData, DataSet
from modelLearning import trainStatefull, testStatefull, trainModel, test
from utils import log, binarize
from evaluation import calculate_metrics, plotScores, plotThreeScores, writeScores
import matplotlib.pyplot as plt
from model import model
from plotFunction import plot
import keras

FMIN = 32.7
HOP_LENGTH = 256
Fs = 22050
HARMONICS = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

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

def plotScores(outPath, preds, labs, realTestSet, testPath):
    # cmap = 'YlOrBr'
    # cmap = "Wistia"
    cmap = "hot"
    if isinstance(preds, list):
        print("Length of scores list:", len(preds))
        for [prediction, target, song] in zip(preds, labs, realTestSet):
            # prediction = binarize(prediction.T)
            print("song:", song)
            if target is not None:
                if isinstance(song, list):
                    song = song[0]
                fig, (ax1, ax2) = plt.subplots(2,1)
                ax1.imshow(target, interpolation='nearest', aspect='auto', cmap=cmap, vmax=1, vmin=0)
                ax1.set_title('Spectrum of Original Song')
                ax2.imshow(prediction, interpolation='nearest', aspect='auto', cmap=cmap, vmax=1, vmin=0)
                ax2.set_title('Estimated Principal Melody Contour')
                plt.savefig(os.path.join(testPath, song[0:-4]+cmap+'.png'))
                plt.close()
    else:
        # prediction = binarize(prediction.T)
        if labs is not None:
            fig, (ax1, ax2) = plt.subplots(2,1)
            ax1.imshow(labs, interpolation='nearest', aspect='auto', cmap=cmap, vmax=1, vmin=0)
            ax1.set_title('Spectrum of Original Song')
            ax2.imshow(preds, interpolation='nearest', aspect='auto', cmap=cmap, vmax=1, vmin=0)
            ax2.set_title('Estimated Principal Melody Contour')
            plt.savefig(os.path.join(testPath, ssong[0:-4]+cmap+'.png'))
            plt.close()

def savePlot(preds, labs, realTestSet, testPath):
    ## SAVE (AND PLOT) TEST RESULTS ###
    p = []
    l = []
    if len(preds[0].shape)==3:
        for (pred, lab) in zip(preds, labs):
            toto = None
            toto2 = None
            for i in range(pred.shape[0]):
                if toto is None:
                    toto = lab[i]
                    toto2 = pred[i]
                else:
                    toto = np.concatenate((toto, lab[i, :, :]), 0)
                    toto2 = np.concatenate((toto2, pred[i, :, :]), 0)
                lim = np.min((toto.shape[0], toto2.shape[0])) # Cut to shortest's length
            labels = toto[:, 1:lim]
            predictions = toto2[:, 1:lim]
            # plt.imshow(predictions.T, aspect='auto')
            # plt.show()
            p.append(predictions)
            l.append(labels)
        plotScores(outPath, p, l, realTestSet, testPath)
    else:
        plotScores(outPath, preds, labs, realTestSet, testPath)

outPath = "/u/anasynth/laffitte/Desktop/moodAgent/"
# deepModel = load_model('/data/anasynth_nonbp/laffitte/Experiments/weights.10-0.733356952667.h5')
deepModel = load_model('melody2')
print(deepModel.summary())
_ = deepModel.predict(np.zeros((2,360,200,6))) # Initialize model

fileList=[]
fileList.append('Antonio Vivaldi - Concerto for 2 Violins in A Minor, Op.3 No. 8, RV 522.mp3')
fileList.append('Jackson 5-The Love I Saw In You Was Just A Mirage.mp3')
fileList.append('Limp Bizkit-Let Me Down.mp3')
predictions = []
original = []
for song in fileList:
    log("Processing track: {}".format(song))
    audioFile = os.path.join("/data/anasynth_nonbp/laffitte", song)
    H = computeHcqt(audioFile, 6, 60, 6)
    curInput = H
    curInput = curInput.transpose(1, 2, 0)[None,:,:,:]
    n_t = curInput.shape[2]
    # Create batches
    output = []
    length = curInput.shape[-1]
    n_slices = 200
    t_slices = list(np.arange(0, n_t, n_slices))
    for i, t in enumerate(t_slices):
        dsRep = deepModel.predict(curInput[:, :, t:t+n_slices, :])
        output.append(dsRep[0,:,:])
    output = np.hstack(output)
    # np.save(os.path.join(pathOut, '{}_mel1_input.npy'.format(f)), output.astype(np.float32))
    output = binarize(output.T).T
    predictions.append(output)
    original.append(curInput[0,:,:,0])

pathFigDsr = outPath
savePlot(predictions, original, fileList, pathFigDsr)

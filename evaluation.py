import numpy as np
import mir_eval
import compute_training_data as C
import scipy
import matplotlib.pyplot as plt
import os, csv, re
from utils import binarize, log
import pandas
import medleydb as mdb
import librosa, pandas

FMIN = 32.7
Fs = 22050#44100

def calculate_metrics(dataobj, preds, labels, inp, trackList, binsPerOctave, nOctave, path, batchSize=500, rnnBatch=16, thresh=0.5, voicing=False):
    log("THRESHOLD IS:", thresh)
    if isinstance(preds, list):
        sequences = False # Data is just a list of songs
    else:
        sequences = True  # Data is formatted in interwoven batches of songs according to statefull generator
    all_scores = []
    inputs = []
    predictions = []
    targets = []
    offset = 0
    if sequences:
        bucketList = dataobj.bucketDataset(trackList, rnnBatch)
        for (b, subTracks) in enumerate(bucketList):
            L, longest = dataobj.findLongest(subTracks[0])
            nSequences = int(np.floor(L/batchSize))
            print(nSequences*rnnBatch, offset, preds.shape)
            for s, song in enumerate(sorted(subTracks[0])):
                ### GET CSV ANNOT FILE TO DETERMINE REAL LENGTH OF SONG
                annotFile = os.path.join('/net/assdb/data/mir2/MedleyDB/Annotations/Melody_Annotations/MELODY2/', song+'_MELODY2.csv')
                if annotFile is not None and os.path.exists(annotFile):
                    annot = pandas.read_csv(annotFile, header=None)
                    f = annot.values[:,1]
                    t = annot.values[:,0]
                    length = int(len(f) / (44100/Fs))
                    ### USE ANNOTATIONS AS LABELS TO COMPUTE SCORES
                    # idx = np.array(np.arange(0, len(t), 44100/Fs), dtype='int32')
                    # ref_freqs = f[idx]
                    # ref_times = t[idx]
                    ref_freqs = None
                    est_freqs = None
                    inputs.append(None)
                    predictions.append(None)
                    targets.append(None)
                    for k in np.arange(s, nSequences*rnnBatch, rnnBatch):
                        pred = preds[offset + k]
                        truth = labels[offset + k]
                        curInput = inp[offset + k]
                        if truth is not None and np.array(truth.nonzero()).any():
                            rt = get_time_grid(binsPerOctave, nOctave, len(truth), Fs, 256)
                            rf = pitch_activations_to_melody(
                            truth, binsPerOctave, nOctave, thresh=thresh, voicing=voicing, mod=False
                            )
                            et = get_time_grid(binsPerOctave, nOctave, len(pred), Fs, 256)
                            ef = pitch_activations_to_melody(
                            pred, binsPerOctave, nOctave, thresh=thresh, voicing=voicing, mod=True
                            )
                            if inputs[-1] is None:
                                ref_freqs = rf
                                est_freqs = ef
                                ref_times = rt
                                est_times = et
                                inputs[-1] = curInput
                                predictions[-1] = pred
                                targets[-1] = truth
                            else:
                                ref_freqs = np.concatenate((ref_freqs, rf))
                                est_freqs = np.concatenate((est_freqs, ef))
                                est_times = np.concatenate((est_times, et))
                                ref_times = np.concatenate((ref_times, rt))
                                inputs[-1] = np.concatenate((inputs[-1], curInput))
                                predictions[-1] = np.concatenate((predictions[-1], pred))
                                targets[-1] = np.concatenate((targets[-1], truth))
                ref_times = ref_times[:length]
                ref_freqs = ref_freqs[:length]
                est_times = est_times[:length]
                est_freqs = est_freqs[:length]
                inputs[-1] = inputs[-1][:length]
                predictions[-1] = predictions[-1][:length]
                targets[-1] = targets[-1][:length]
                scores = mir_eval.melody.evaluate(ref_times, ref_freqs, est_times, est_freqs)
                all_scores.append(scores)

                csvPath = os.path.join(path, 'csv')
                if not os.path.exists(csvPath):
                    os.mkdir(csvPath)
                with open(os.path.join(csvPath, song +'referenceHertz.csv'), 'w') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerows([ref_freqs, ref_times])
                with open(os.path.join(csvPath, song +'predictionHertz.csv'), 'w') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerows([est_freqs, est_times])
                np.save(os.path.join(path, song+'preds.npy'), predictions[-1])
                np.save(os.path.join(path, song+'targets.npy'), targets[-1])
            offset += k+1
        # predictions = binarize(predictions)
        plotScores(predictions, targets, inputs, trackList, path, rnnBatch)
    else:
        for (pred, truth, curInput, song) in zip(preds, labels, inp, trackList):
            if truth is not None and np.array(truth.nonzero()).any():
                if len(truth)==len(pred)-1:
                    pred = pred[:-1,:]
                mask = np.where(truth==-1)
                if any(mask[1]):
                    curInput = curInput[0:mask[0][0],:]
                    truth = truth[0:mask[0][0],:]
                    pred = pred[0:mask[0][0],:]
                ref_times = get_time_grid(binsPerOctave, nOctave, len(truth), Fs, 256)
                ref_freqs = pitch_activations_to_melody(
                truth, binsPerOctave, nOctave, thresh=thresh, voicing=voicing, mod=False
                )
                est_times = get_time_grid(binsPerOctave, nOctave, len(pred), Fs, 256)
                est_freqs = pitch_activations_to_melody(
                pred, binsPerOctave, nOctave, thresh=thresh, voicing=voicing, mod=True
                )
                ### GET CSV ANNOT FILE TO DETERMINE REAL LENGTH OF SONG
                annotFile = os.path.join('/net/assdb/data/mir2/MedleyDB/Annotations/Melody_Annotations/MELODY2/', song+'_MELODY2.csv')
                if annotFile is not None and os.path.exists(annotFile):
                    annot = pandas.read_csv(annotFile, header=None)
                    f = annot.values[:,1]
                    t = annot.values[:,0]
                    length = int(len(f) / (44100/Fs))
                    est_times = est_times[:length]
                    est_freqs = est_freqs[:length]
                    ref_times = ref_times[:length]
                    ref_freqs = ref_freqs[:length]
                    np.save(os.path.join(path, song+'preds.npy'), pred)
                    np.save(os.path.join(path, song+'targets.npy'), truth)
                    scores = mir_eval.melody.evaluate(ref_times, ref_freqs, est_times, est_freqs)
                    all_scores.append(scores)

                    csvPath = os.path.join(path, 'csv')
                    if not os.path.exists(csvPath):
                        os.mkdir(csvPath)
                    with open(os.path.join(csvPath, song +'referenceHertz.csv'), 'w') as f:
                        writer = csv.writer(f, delimiter=',')
                        writer.writerows([ref_freqs, ref_times])
                    with open(os.path.join(csvPath, song +'predictionHertz.csv'), 'w') as f:
                        writer = csv.writer(f, delimiter=',')
                        writer.writerows([est_freqs, est_times])
        # preds = binarize(preds)
        plotScores(preds, labels, inp, trackList, path, rnnBatch)
    return all_scores, [], [], inputs

def pitch_activations_to_melody(pitch_activation_mat, binsPerOctave, nOctave, thresh=0.5, voicing=False, mod=False):
    """Convert a pitch activation map to melody line (sequence of frequencies)
    """
    s = pitch_activation_mat.shape
    if s[0]==binsPerOctave*nOctave or s[0]==binsPerOctave*nOctave+1:
        ind = 0 # index of frequency axis
    else:
        ind = 1
    freqs = get_freq_grid(binsPerOctave, nOctave)
    if voicing:
        melodyEstimation = np.zeros((s[1-ind])) # build melody pitch estimation vector
        voiced = np.zeros(s[1-ind]) # build voicing array
        highest = np.argmax(pitch_activation_mat, ind)
        idxThreshold = np.where(highest==0)[0] # get voicing predictions
        voiced[idxThreshold] = 1
        voiced = 1 - voiced
        estFreqs = freqs[highest-1] # read wich frequencies were detected
        secondHighest = np.argmax(pitch_activation_mat[:,1:], ind)
        estUnvoiced = freqs[(secondHighest)] # read second highest probability for unvoiced
        melodyEstimation[voiced==1] = estFreqs[voiced==1]
        melodyEstimation[voiced==0] = 0 - estUnvoiced[voiced==0]
        if mod:
            if s[0]==binsPerOctave*nOctave or s[0]==binsPerOctave*nOctave+1:
                pitch_activation_mat[:,voiced==0] = 0
            else:
                pitch_activation_mat[voiced==0,:] = 0
    else:
        max_idx = np.argmax(pitch_activation_mat, axis=ind)
        melodyEstimation = []
        nopitch = np.where(pitch_activation_mat.any(ind)==False)[0]
        for i, f in enumerate(max_idx):
            if i in nopitch:
                melodyEstimation.append(0.0)
            else:
                if pitch_activation_mat[i, f] < thresh and mod:
                    melodyEstimation.append(-1.0*freqs[f])
                else:
                    melodyEstimation.append(freqs[f])
        melodyEstimation = np.array(melodyEstimation)

    return melodyEstimation



def get_time_grid(bins_per_octave, n_octaves, n_time_frames, Fs, hop):
    """Get the hcqt time grid
    """
    time_grid = librosa.core.frames_to_time(
        range(n_time_frames), sr=Fs, hop_length=hop
    )
    return time_grid

def get_freq_grid(bins_per_octave, n_octaves):
    """Get the hcqt frequency grid
    """
    freq_grid = librosa.cqt_frequencies(
        bins_per_octave*n_octaves, FMIN, bins_per_octave=bins_per_octave
    )
    return freq_grid

def plotThreeScores(preds, labs, cnn, all_scores, realTestSet, testPath):
    for [prediction, target, cnnOut, song] in zip(preds, labs, cnn, realTestSet):
        prediction = binarize(prediction)
        fig, (ax1, ax2, ax3) = plt.subplots(3,1)
        ax1.imshow(target.T, interpolation='nearest', aspect='auto', vmax=1, vmin=0)
        ax1.set_title('Labels')
        ax2.imshow(cnnOut.T, interpolation='nearest', aspect='auto', vmax=1, vmin=0)
        ax2.set_title('CNN Output')
        ax3.imshow(prediction.T, interpolation='nearest', aspect='auto')
        ax3.set_title('RNN Output')
        plt.savefig(os.path.join(testPath, song[0]+'_result_melody1.png'))
        plt.close()

def plotScores(preds, labs, inputs, realTestSet, testPath, rnnBatch):
    cmap = "viridis"
    if isinstance(preds, list):
        plotLinear(preds, labs, inputs, realTestSet, testPath)
    else:
        preds = binarize(preds)
        if labs is not None:
            pred = [None] * rnnBatch
            inp = [None] * rnnBatch
            lab = [None] * rnnBatch
            for j in range(preds.shape[0]):
                for k in range(rnnBatch):
                    if inp[k] is None:
                        inp[k] = inputs[j]
                        pred[k] = preds[j]
                        lab[k] = labs[j]
                    else:
                        inp[k] = np.concatenate((inp[k], inputs[j]), 0)
                        pred[k] = np.concatenate((pred[k], preds[j]), 0)
                        lab[k] = np.concatenate((lab[k], labs[j]), 0)
            for k in range(rnnBatch):
                fig, (ax1, ax2, ax3) = plt.subplots(3,1)
                ax1.imshow(inp[k].T, aspect='auto', cmap="hot", vmax=1, vmin=0, origin='lower')
                ax1.set_title('INPUTS')
                ax2.imshow(pred[k].T, aspect='auto', cmap="hot", vmax=1, vmin=0, origin='lower')
                ax2.set_title('OUTPUT')
                ax3.imshow(lab[k].T, aspect='auto', cmap="hot", vmax=1, vmin=0, origin='lower')
                ax3.set_title('TARGETS')
                plt.savefig(os.path.join(testPath, 'batch-{}_result_melody1.png'.format(k)))
                plt.close()
                np.save(os.path.join(testPath, 'mel2_targets.npy'), lab[k].astype(np.float32))
                np.save(os.path.join(testPath, 'mel2_outputs.npy'), pred[k].astype(np.float32))

def plotLinear(output, label, inputs, songList, testPath):
    cmap = "viridis"
    for [out, lab, inp, song] in zip(output, label, inputs, songList):
        if out.shape[0] > out.shape[1]:
            out = out.transpose(1,0)
        if lab.shape[0] > lab.shape[1]:
            lab = lab.transpose(1,0)
        if len(inp.shape)==3:
            inp = inp[0,:,:]
        else:
            inp = inp
        # plt.imshow(lab, aspect='auto', origin='lower', cmap=cmap, vmax=1, vmin=0)
        # fig, (ax1, ax2, ax3) = plt.subplots(3,1)
        fig, (ax1, ax2) = plt.subplots(2,1)
        # ax3.imshow(inp.T, aspect='auto', origin='lower', cmap=cmap, vmax=1, vmin=0)
        # ax3.set_title('INPUTS')
        ax2.imshow(lab, aspect='auto', origin='lower', cmap=cmap, vmax=1, vmin=0)
        ax2.set_title('TARGETS')
        ax1.imshow(out, aspect='auto', origin='lower', cmap=cmap, vmax=1, vmin=0)
        ax1.set_title('OUTPUTS')
        plt.savefig(os.path.join(testPath, song+'_result_melody1.png'))
        plt.close()
        # np.save(os.path.join(testPath, 'mel2_targets.npy'), lab.astype(np.float32))
        # np.save(os.path.join(testPath, 'mel2_outputs.npy'), out.astype(np.float32))

def writeScores(all_scores, outPath):
    log("Writing scores to csv files")
    scores_path = os.path.join(outPath, 'all_scores.csv')
    score_summary_path = os.path.join(outPath, "score_summary.csv")
    df = pandas.DataFrame(data=all_scores)
    df.to_csv(scores_path)
    df.describe().to_csv(score_summary_path)
    print(df.describe())

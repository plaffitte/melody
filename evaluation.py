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

def calculate_metrics(dataobj, preds, labels, trackList, binsPerOctave, nOctave, path, batchSize=500, rnnBatch=16, thresh=0.5, deepsalience=False):
    log("THRESHOLD IS:", thresh)
    if isinstance(preds, list):
        sequences = False # Data is just a list of songs
    else:
        sequences = True  # Data is formatted in interwoven batches of songs
    all_scores = []
    predictions = []
    targets = []
    offset = 0
    # if sequences:
    bucketList = dataobj.bucketDataset(trackList, rnnBatch)
    for (b, subTracks) in enumerate(bucketList):
        L, longest = dataobj.findLongest(subTracks[0])
        for s, song in enumerate(sorted(subTracks[0])):
            ### GET CSV ANNOT FILE TO DETERMINE REAL LENGTH OF SONG
            annotFile = os.path.join('/net/assdb/data/mir2/MedleyDB/Annotations/Melody_Annotations/MELODY2/', song+'_MELODY2.csv')
            if annotFile is not None and os.path.exists(annotFile):
                annot = pandas.read_csv(annotFile, header=None)
                f = annot.values[:,1]
                t = annot.values[:,0]
                idx = int(44100/Fs) ### ANNOT FILES HAVE a 5ms HOP SIZE !!!
                f = f[::idx]
                ref_freqs = None
                est_freqs = None
                predictions.append(None)
                targets.append(None)
                nSequences = int(np.ceil(len(f)/batchSize))
                for k in np.arange(s, nSequences*rnnBatch, rnnBatch):
                    if (offset+k) < preds.shape[0]:
                        pred = preds[offset + k]
                        truth = labels[offset + k]
                        rf = pitch_activations_to_melody(
                        truth, nOctave, thresh=thresh, deepsalience=deepsalience, mod=False
                        )
                        ef = pitch_activations_to_melody(
                        pred, nOctave, thresh=thresh, deepsalience=deepsalience, mod=True
                        )
                        if est_freqs is None:
                            ref_freqs = rf
                            est_freqs = ef
                            predictions[-1] = pred
                            targets[-1] = truth
                        else:
                            ref_freqs = np.concatenate((ref_freqs, rf))
                            est_freqs = np.concatenate((est_freqs, ef))
                            predictions[-1] = np.concatenate((predictions[-1], pred))
                            targets[-1] = np.concatenate((targets[-1], truth))
                ref_times = get_time_grid(binsPerOctave, nOctave, len(ref_freqs), Fs, 256)
                est_times = get_time_grid(binsPerOctave, nOctave, len(f), Fs, 256)
                length = np.min((len(f), len(est_freqs)))
                ref_times = ref_times[:length]
                ref_freqs = ref_freqs[:length]
                est_times = est_times[:length]
                est_freqs = est_freqs[:length]
                predictions[-1] = predictions[-1][:length]
                targets[-1] = targets[-1][:length]
                f = f[:length]
                time_index = np.arange(length) * 256./22050
                # scores = mir_eval.melody.evaluate(ref_times, ref_freqs, est_times, est_freqs)
                scores = mir_eval.melody.evaluate(time_index, f, time_index, est_freqs)
                all_scores.append(scores)
                csvPath = os.path.join(path, 'csv')
                if not os.path.exists(csvPath):
                    os.mkdir(csvPath)
                with open(os.path.join(csvPath, song +'referenceHertz.csv'), 'w') as fi:
                    writer = csv.writer(fi, delimiter=',')
                    writer.writerows(np.concatenate([time_index[:,None], f[:,None]], 1))
                    # writer.writerows(np.concatenate([ref_times[:,None], ref_freqs[:,None]], 1))
                with open(os.path.join(csvPath, song +'.csv'), 'w') as fi:
                    writer = csv.writer(fi, delimiter=',')
                    writer.writerows(np.concatenate([time_index[:,None], est_freqs[:,None]], 1))
                    # writer.writerows(np.concatenate([est_times[:,None], est_freqs[:,None]], 1))
        offset += int(int(np.floor(L/batchSize))*rnnBatch)
    predictions = binarize(predictions)
    plotScores(predictions, targets, trackList, path, rnnBatch)
    # else:
        # for (pred, truth, curInput, song) in zip(preds, labels, inp, trackList):
        #     if truth is not None and np.array(truth.nonzero()).any():
        #         if len(truth)==len(pred)-1:
        #             pred = pred[:-1,:]
        #         mask = np.where(truth==-1)
        #         if any(mask[1]):
        #             curInput = curInput[0:mask[0][0],:]
        #             truth = truth[0:mask[0][0],:]
        #             pred = pred[0:mask[0][0],:]
        #         ref_times = get_time_grid(binsPerOctave, nOctave, len(truth), Fs, 256)
        #         ref_freqs = pitch_activations_to_melody(
        #         truth, nOctave, thresh=thresh, deepsalience=False, mod=False
        #         )
        #         est_times = get_time_grid(binsPerOctave, nOctave, len(pred), Fs, 256)
        #         est_freqs = pitch_activations_to_melody(
        #         pred, nOctave, thresh=thresh, deepsalience=deepsalience, mod=True
        #         )
        #         ### GET CSV ANNOT FILE TO DETERMINE REAL LENGTH OF SONG
        #         annotFile = os.path.join('/net/assdb/data/mir2/MedleyDB/Annotations/Melody_Annotations/MELODY2/', song+'_MELODY2.csv')
        #         if annotFile is not None and os.path.exists(annotFile):
        #             annot = pandas.read_csv(annotFile, header=None)
        #             f = annot.values[:,1]
        #             t = annot.values[:,0]
        #             idx = int(44100/Fs) ### ANNOT FILES HAVE a 5ms HOP SIZE !!!
        #             f = f[::idx]
        #             length = np.min((len(f), len(est_freqs)))
        #             # est_times = est_times[:length]
        #             est_freqs = est_freqs[:length]
        #             # ref_times = ref_times[:length]
        #             ref_freqs = ref_freqs[:length]
        #             f = f[:length]
        #             time_index = np.arange(length) * 256./Fs
        #             # scores = mir_eval.melody.evaluate(time_index, ref_freqs, time_index, est_freqs)
        #             scores = mir_eval.melody.evaluate(time_index, f, time_index, est_freqs)
        #             all_scores.append(scores)
        #             csvPath = os.path.join(path, 'csv')
        #             if not os.path.exists(csvPath):
        #                 os.mkdir(csvPath)
        #             with open(os.path.join(csvPath, song +'referenceHertz.csv'), 'w') as fi:
        #                 writer = csv.writer(fi, delimiter=',')
        #                 writer.writerows(np.concatenate([time_index[:,None], ref_freqs[:,None]], 1))
        #             with open(os.path.join(csvPath, song +'.csv'), 'w') as fi:
        #                 writer = csv.writer(fi, delimiter=',')
        #                 writer.writerows(np.concatenate([time_index[:,None], est_freqs[:,None]], 1))
        # preds = binarize(preds)
        # plotScores(preds, labels, inp, trackList, path, rnnBatch)
    return all_scores, [], []

def computeDeepSalienceScores(dataobj, preds, trackList, binsPerOctave, nOctave, path, batchSize=500, rnnBatch=16, thresh=0.3):
    log("THRESHOLD IS:", thresh)
    all_scores = []
    preds = reshapePredictions(dataobj, preds, trackList, rnnBatch, batchSize)
    for (pred, song) in zip(preds, trackList):
        est_times = get_time_grid(binsPerOctave, nOctave, len(pred), Fs, 256)
        est_freqs = pitch_activations_to_melody(
        pred, nOctave, thresh=thresh, deepsalience=True, mod=True
        )
        ### GET CSV ANNOT FILE TO DETERMINE REAL LENGTH OF SONG
        annotFile = os.path.join('/net/assdb/data/mir2/MedleyDB/Annotations/Melody_Annotations/MELODY2/', song+'_MELODY2.csv')
        if annotFile is not None and os.path.exists(annotFile):
            annot = pandas.read_csv(annotFile, header=None)
            f = annot.values[:,1]
            t = annot.values[:,0]
            idx = int(44100/Fs) ### ANNOT FILES HAVE a 5ms HOP SIZE !!!
            f = f[::idx]
            length = np.min((len(f), len(est_freqs)))
            est_freqs = est_freqs[:length]
            f = f[:length]
            time_index = np.arange(length) * 256./Fs
            scores = mir_eval.melody.evaluate(time_index, f, time_index, est_freqs)
            all_scores.append(scores)
            csvPath = os.path.join(path, 'csv')
            if not os.path.exists(csvPath):
                os.mkdir(csvPath)
            with open(os.path.join(csvPath, song +'referenceHertz.csv'), 'w') as fi:
                writer = csv.writer(fi, delimiter=',')
                writer.writerows(np.concatenate([time_index[:,None], f[:,None]], 1))
            with open(os.path.join(csvPath, song +'.csv'), 'w') as fi:
                writer = csv.writer(fi, delimiter=',')
                writer.writerows(np.concatenate([time_index[:,None], est_freqs[:,None]], 1))
        pred = binarize(pred)
        # plotScores(pred, trackList, path, rnnBatch)

    return all_scores

def reshapePredictions(dataobj, preds, trackList, rnnBatch, batchSize):
    """ Turn array of interwoven data into list, "sorted" by song.
    """
    predictions = []
    offset = 0
    bucketList = dataobj.bucketDataset(trackList, rnnBatch)
    for (b, subTracks) in enumerate(bucketList):
        length, _ = dataobj.findLongest(subTracks[0])
        for s, song in enumerate(sorted(subTracks[0])):
            L, longest = dataobj.findLongest([song])
            nSequences = int(np.ceil(L/batchSize))
            predictions.append(None)
            for k in np.arange(s, nSequences*rnnBatch, rnnBatch):
                if (offset+k) < preds.shape[0]:
                    pred = preds[offset + k]
                    if predictions[-1] is None:
                        predictions[-1] = pred
                    else:
                        predictions[-1] = np.concatenate((predictions[-1], pred))
        offset += int(int(np.floor(length/batchSize))*rnnBatch)

    return predictions

def pitch_activations_to_melody(pitch_activation_mat, nOctave, thresh=0.5,  deepsalience=False, mod=False):
    """Convert a pitch activation map to melody line (sequence of frequencies)
    """
    s = pitch_activation_mat.shape
    if s[0]==72 or s[0]==73 or s[0]==360 or s[0]==361:
        ind = 0 # index of frequency axis
    else:
        ind = 1
    if not deepsalience:
        freqs = [0]
        freqs.extend(librosa.cqt_frequencies(s[ind], FMIN, s[ind]/nOctave)[:-1])
        freqs = np.array(freqs)
        melodyEstimation = np.zeros((s[1-ind])) # build melody pitch estimation vector
        voiced = np.zeros(s[1-ind]) # build voicing array
        highest = np.argmax(pitch_activation_mat, ind)
        idxVoiced = np.where(highest==0)[0] # get voicing predictions
        voiced[idxVoiced] = 1
        voiced = 1 - voiced
        melodyEstimation = freqs[highest.astype(np.int)]
        ### For frames estimated as unvoiced, put negative value of second highest frequency detected as required by mir_eval metrics
        if ind==0:
            newPitchMat = pitch_activation_mat[1:,:]
        else:
            newPitchMat = pitch_activation_mat[:,1:]
        secondHighest = np.argmax(newPitchMat, ind)
        estUnvoiced = 0 - freqs[secondHighest+1] # read second highest probability for unvoiced frames
        if mod:
            melodyEstimation[voiced==0] = estUnvoiced[voiced==0]
        else:
            melodyEstimation[voiced==0] = 0
        if mod:
            if ind == 0:
                pitch_activation_mat[:,voiced==0] = 0
            else:
                pitch_activation_mat[voiced==0,:] = 0
    else:
        freqs = librosa.cqt_frequencies(s[ind], FMIN, s[ind]/nOctave)
        freqs = np.array(freqs)
        max_idx = np.argmax(pitch_activation_mat, axis=ind)
        melodyEstimation = []
        pitch = np.where(pitch_activation_mat>=thresh)[1-ind]
        for i, f in enumerate(max_idx):
            if i not in pitch:
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

def plotThreeScores(preds, labs, cnn, all_scores, realTestSet, testPath):
    for [prediction, target, cnnOut, song] in zip(preds, labs, cnn, realTestSet):
        prediction = binarize(prediction)
        cnnOut = binarize(cnnOut)
        fig, (ax1, ax2, ax3) = plt.subplots(3,1)
        ax1.imshow(target.T, interpolation='nearest', aspect='auto', vmax=1, vmin=0)
        ax1.set_title('Labels')
        ax2.imshow(cnnOut.T, interpolation='nearest', aspect='auto', vmax=1, vmin=0)
        ax2.set_title('CNN Output')
        ax3.imshow(prediction.T, interpolation='nearest', aspect='auto')
        ax3.set_title('RNN Output')
        plt.savefig(os.path.join(testPath, song[0]+'_result_melody2.png'))
        plt.close()

def plotScores(preds, labs, realTestSet, testPath, rnnBatch):
    cmap = "viridis"
    if isinstance(preds, list):
        plotLinear(preds, labs, realTestSet, testPath)
    else:
        preds = binarize(preds)
        if labs is not None:
            pred = [None] * rnnBatch
            lab = [None] * rnnBatch
            for j in range(preds.shape[0]):
                for k in range(rnnBatch):
                    if inp[k] is None:
                        pred[k] = preds[j]
                        lab[k] = labs[j]
                    else:
                        pred[k] = np.concatenate((pred[k], preds[j]), 0)
                        lab[k] = np.concatenate((lab[k], labs[j]), 0)
            for k in range(rnnBatch):
                fig, (ax1, ax2) = plt.subplots(2,1)
                ax1.imshow(pred[k].T, aspect='auto', cmap="hot", vmax=1, vmin=0, origin='lower')
                ax1.set_title('OUTPUT')
                ax2.imshow(lab[k].T, aspect='auto', cmap="hot", vmax=1, vmin=0, origin='lower')
                ax2.set_title('TARGETS')
                plt.savefig(os.path.join(testPath, 'batch-{}_result_melody2.png'.format(k)))
                plt.close()

def plotLinear(output, label, songList, testPath):
    cmap = "viridis"
    for [out, lab, song] in zip(output, label, songList):
        if out.shape[0] > out.shape[1]:
            out = out.transpose(1,0)
        if lab.shape[0] > lab.shape[1]:
            lab = lab.transpose(1,0)
        fig, (ax1, ax2) = plt.subplots(2,1)
        ax2.imshow(lab, aspect='auto', origin='lower', cmap=cmap, vmax=1, vmin=0)
        ax2.set_title('TARGETS')
        ax1.imshow(out, aspect='auto', origin='lower', cmap=cmap, vmax=1, vmin=0)
        ax1.set_title('OUTPUTS')
        plt.show()
        plt.savefig(os.path.join(testPath, song+'_result_melody2.png'))
        plt.close()

def writeScores(all_scores, outPath):
    log("Writing scores to csv files")
    scores_path = os.path.join(outPath, 'all_scores.csv')
    score_summary_path = os.path.join(outPath, "score_summary.csv")
    df = pandas.DataFrame(data=all_scores)
    df.to_csv(scores_path)
    df.describe().to_csv(score_summary_path)
    print(df.describe())

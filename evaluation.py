import numpy as np
import mir_eval
import compute_training_data as C
import scipy
import matplotlib.pyplot as plt
import os, csv
from utils import binarize, log

def calculate_metrics(preds, labels, trackList, binsPerOctave, nOctave, thresh=0.5, voicing=False):

    log("THRESHOLD IS:", thresh)
    if len(preds[0].shape)>=3:
        sequences = True
    else:
        sequences = False
    all_scores = []
    melodyEstimation = []
    for [pred, truth, track] in zip(preds, labels, trackList):
        if sequences:
            if truth is not None:
                ref_times = None
                ref_freqs = None
                for i in range(len(pred)):
                    seq = pred[i]
                    base = truth[i]
                    rt, rf = pitch_activations_to_melody(
                    base, binsPerOctave, nOctave, thresh=thresh, voicing=voicing, mod=False
                    )
                    et, ef = pitch_activations_to_melody(
                    seq, binsPerOctave, nOctave, thresh=thresh, voicing=voicing, mod=True
                    )
                    if ref_freqs is None:
                        ref_times, ref_freqs = rt, rf
                        est_times, est_freqs = et, ef
                    else:
                        ref_times = np.concatenate((ref_times, rt))
                        ref_freqs = np.concatenate((ref_freqs, rf))
                        est_times = np.concatenate((est_times, et))
                        est_freqs = np.concatenate((est_freqs, ef))
        else:
            if truth is not None:
                ref_times, ref_freqs = pitch_activations_to_melody(
                truth, binsPerOctave, nOctave, thresh=thresh, voicing=voicing, mod=False
                )
                est_times, est_freqs = pitch_activations_to_melody(
                pred, binsPerOctave, nOctave, thresh=thresh, voicing=voicing, mod=True
                )
        # convert time and freq arrays to boolean and cents
        # ref_voicing, ref_cent, est_voicing, est_cent = mir_eval.melody.to_cent_voicing(ref_times, ref_freqs, est_times, est_freqs)
        scores = mir_eval.melody.evaluate(ref_times, ref_freqs, est_times, est_freqs)
        all_scores.append(scores)
        melodyEstimation.append(ref_freqs)
    return all_scores, melodyEstimation

def get_best_thresh(dat, model, files):

    thresh_vals = np.arange(0.1, 1.0, 0.1)
    thresh_scores = {t: [] for t in thresh_vals}
    for track in files:
        preds = glob.glob(os.path.join(self.input_path, '{}_mel1_input.npy'.format(track)))
        truth = glob.glob(os.path.join(self.input_path, '{}_mel1_input.npy'.format(track)))
        curInput = np.load(track[0])
        file_keys = os.path.basename(track).split('_')[:2]
        label_file = glob.glob(
            os.path.join(
                test_set_path, 'mdb_test',
                "{}*{}.txt".format(file_keys[0], file_keys[1]))
        )[0]

        # generate prediction on nupitch_activation_matmpy file
        predicted_output, input_hcqt = \
            get_single_test_prediction(curInput, model)

        # load ground truth labels
        ref_times, ref_freqs = \
            mir_eval.io.load_ragged_time_series(label_file)

        for thresh in thresh_vals:
            # get multif0 output from prediction
            est_times, est_freqs = \
                pitch_activations_to_mf0(predicted_output, thresh)

            # get multif0 metrics and append
            scores = mir_eval.multipitch.evaluate(
                ref_times, ref_freqs, est_times, est_freqs)
            thresh_scores[thresh].append(scores['Accuracy'])

    return thresh_vals, thresh_scores

def pitch_activations_to_melody(pitch_activation_mat, binsPerOctave, nOctave, thresh=0.5, voicing=False, mod=False):
    """Convert a pitch activation map to melody line (sequence of frequencies)
    """
    s = pitch_activation_mat.shape
    if s[0]==binsPerOctave*nOctave or s[0]==binsPerOctave*nOctave+1:
        ind = 0
    else:
        ind = 1
    freqs = C.get_freq_grid(binsPerOctave, nOctave)
    times = C.get_time_grid(binsPerOctave, nOctave, s[ind])
    if voicing:
        melodyEstimation = np.zeros((s[1-ind])) # build melody pitch estimation vector
        voiced = np.zeros(s[1-ind]) # build voicing array
        highest = np.argmax(pitch_activation_mat, ind)
        idxThreshold = np.where(highest==0)[0] # get voicing predictions
        voiced[idxThreshold] = 1
        voiced = 1 - voiced
        estFreqs = freqs[highest] # read wich frequencies were detected
        if s[0]==binsPerOctave*nOctave or s[0]==binsPerOctave*nOctave+1:
            newPitchMat = np.zeros((s[ind] - 1, s[1-ind]))
        else:
            newPitchMat = np.zeros((s[1-ind], s[ind] - 1))
        for i in range(len(highest)):
            if s[0]==binsPerOctave*nOctave:
                newPitchMat[:,i] = np.delete(pitch_activation_mat[:,i], highest[i])
            else:
                newPitchMat[i,:] = np.delete(pitch_activation_mat[i,:], highest[i])
        secondHighest = np.argmax(newPitchMat, ind)
        estVoiced = freqs[(secondHighest + 1)] # read second highest probability for unvoiced
        melodyEstimation[voiced==1] = estFreqs[voiced==1]
        if mod:
            if s[0]==binsPerOctave*nOctave or s[0]==binsPerOctave*nOctave+1:
                pitch_activation_mat[:,voiced==0] = 0
            else:
                pitch_activation_mat[voiced==0,:] = 0
    else:
        melodyEstimation = np.zeros((s[1-ind])) # build melody pitch estimation vector
        voiced = np.zeros(s[1-ind]) # build voicing array
        highest = np.argmax(pitch_activation_mat, ind)
        idxThreshold = np.where(highest >= thresh) # get time index where predictions are above threshold
        voiced[idxThreshold] = 1
        if s[0]==binsPerOctave*nOctave or s[0]==binsPerOctave*nOctave+1:
            newPitchMat = np.zeros((s[ind] - 1, s[1-ind]))
        else:
            newPitchMat = np.zeros((s[1-ind], s[ind] - 1))
        for i in range(len(highest)):
            if s[0]==binsPerOctave*nOctave:
                newPitchMat[:,i] = np.delete(pitch_activation_mat[:,i], highest[i])
            else:
                newPitchMat[i,:] = np.delete(pitch_activation_mat[i,:], highest[i])
        secondHighest = np.argmax(newPitchMat, ind)
        estFreqs = freqs[highest] # read which frequencies were detected
        estVoiced = freqs[(secondHighest + 1)] # read second highest probability for unvoiced
        melodyEstimation[voiced==1] = estFreqs[voiced==1]
        if mod:
            melodyEstimation[voiced==0] = 0 - estVoiced[voiced==0]
            if s[0]==binsPerOctave*nOctave or s[0]==binsPerOctave*nOctave+1:
                pitch_activation_mat[:,voiced==0] = 0
            else:
                pitch_activation_mat[voiced==0,:] = 0

    return times, melodyEstimation

def plotThreeScores(preds, labs, cnn, all_scores, realTestSet, testPath):
    for [prediction, target, cnnOut, song] in zip(preds, labs, cnn, realTestSet):
        # prediction = binarize(prediction)
        fig, (ax1, ax2, ax3) = plt.subplots(3,1)
        ax1.imshow(target.T, interpolation='nearest', aspect='auto', vmax=1, vmin=0)
        ax1.set_title('Labels')
        ax2.imshow(cnnOut.T, interpolation='nearest', aspect='auto', vmax=1, vmin=0)
        ax2.set_title('CNN Output')
        ax3.imshow(prediction.T, interpolation='nearest', aspect='auto')
        ax3.set_title('RNN Output')
        plt.savefig(os.path.join(testPath, song[0]+'_result_melody1.png'))
        plt.close()

def plotScores(preds, labs, inputs, realTestSet, testPath):
    cmap = "hot"
    print(len(preds, preds[0].shape))
    if isinstance(preds, list):
        for [prediction, target, inp, song] in zip(preds, labs, inputs, realTestSet):
            # prediction = binarize(prediction)
            log("song:", song)
            if target is not None:
                if target.shape[0] > target.shape[1]:
                    target = target.T
                if prediction.shape[0] > prediction.shape[1]:
                    prediction = prediction.T
                if inp.shape[0] > inp.shape[1]:
                    inp = inp.T
                if isinstance(song, list):
                    song = song[0]
                mask = inp.nonzero()
                print(len(realInput))
                print(mask[0][-1])
                if any(mask[0]):
                    inp = realInput[0:mask[0][-1],:]
                    tar = realLabel[0:mask[0][-1],:]
                    pred = realPred[0:mask[0][-1],:]
                fig, (ax1, ax2, ax3) = plt.subplots(3,1)
                ax1.imshow(inp, aspect='auto', cmap=cmap, vmax=1, vmin=0)
                ax1.set_title('INPUTS')
                ax2.imshow(tar, interpolation='nearest', aspect='auto', cmap=cmap, vmax=1, vmin=0)
                ax2.set_title('TARGETS')
                ax3.imshow(pred, interpolation='nearest', aspect='auto', cmap=cmap, vmax=1, vmin=0)
                ax3.set_title('OUTPUTS')
                plt.savefig(os.path.join(testPath, song+'_result_melody1.png'))
                plt.close()
    else:
        # prediction = binarize(prediction)
        if labs is not None:
            print(len(labs))
            print(len(preds))
            fig, (ax1, ax2) = plt.subplots(2,1)
            ax1.imshow(labs.T, interpolation='nearest', aspect='auto', cmap=cmap, vmax=1, vmin=0)
            ax1.set_title('Labels')
            ax2.imshow(preds.T, interpolation='nearest', aspect='auto', cmap=cmap, vmax=1, vmin=0)
            ax2.set_title('Output')
            plt.savefig(os.path.join(testPath, song[0]+'_result_melody1.png'))
            plt.close()

def arrangePredictions(preds, labs, inputs, dataSet, path, cnnOut=[]):
    ## SAVE (AND PLOT) TEST RESULTS ###
    p = []
    l = []
    c = []
    i = []
    print(len(preds), preds[0].shape)
    if len(preds[0].shape)==3:
        if "TESTDEEPSALIENCE" in modelDim:
            for (pred, lab, inp) in zip(preds, labs, inputs):
                toto = np.zeros((1, int(fftSize)))
                toto2 = np.zeros((1, int(fftSize)))
                toto3 = np.zeros((1, int(fftSize)))
                for i in range(pred.shape[0]):
                    toto = np.concatenate((toto, lab[i, :, :]), 0)
                    toto2 = np.concatenate((toto2, pred[i, :, :]), 0)
                    toto3 = np.concatenate((toto3, inp[i, :, :]), 0)
                    lim = np.min((toto.shape[-1], toto2.shape[-1])) # Cut to shortest's length
                labels = toto[:, 1:lim]
                predictions = toto2[:, 1:lim]
                curInput = toto3[:, 1:lim]
                p.append(predictions)
                l.append(labels)
                i.append(curInput)
        else:
            for (pred, lab, cnn, inp) in zip(preds, labs, inputs, cnnOut):
                # pred = binarize(pred)
                if "SOFTMAX" in modelDim or "VOICING" in modelDim:
                    toto = np.zeros((int(fftSize)+1, 1))
                    toto2 = np.zeros((int(fftSize)+1, 1))
                    toto3 = np.zeros((int(fftSize)+1, 1))
                    toto4 = np.zeros((int(fftSize)+1, 1))
                else:
                    toto = np.zeros((int(fftSize), 1))
                    toto2 = np.zeros((int(fftSize), 1))
                    toto3 = np.zeros((int(fftSize), 1))
                    toto4 = np.zeros((int(fftSize), 1))
                for i in range(pred.shape[0]):
                    for j in range(pred.shape[2]):
                        toto = np.concatenate((toto, lab[i, :, j, None]), 1)
                        toto2 = np.concatenate((toto2, pred[i, :, j, None]), 1)
                        toto3 = np.concatenate((toto3, cnn[i, :, j, None]), 1)
                        toto4 = np.concatenate((toto4, inp[i, :, j, None]), 1)
                        lim = np.min((toto.shape[-1], toto2.shape[-1])) # Cut to shortest's length
                labels = toto[:, 1:lim]
                predictions = toto2[:, 1:lim]
                cnnOutput = toto3[:, 1:lim]
                curInput = toto4[:, 1:lim]
                p.append(predictions)
                l.append(labels)
                i.append(curInput)
                c.append(cnnOutput)
        plotScores(p, l, i, dataSet, path)
    else:
        plotScores(preds, labs, inputs, dataSet, path)

def writeScores(all_scores, outPath):
    meanScores = {}
    for k in all_scores[0].keys():
        meanScores[k] = []
    for score in all_scores:
        meanScores['Overall Accuracy'].append(score['Overall Accuracy'])
        meanScores['Raw Pitch Accuracy'].append(score['Raw Pitch Accuracy'])
        meanScores['Raw Chroma Accuracy'].append(score['Raw Chroma Accuracy'])
        meanScores['Voicing Recall'].append(score['Voicing Recall'])
        meanScores['Voicing False Alarm'].append(score['Voicing False Alarm'])
    meanScores['Overall Accuracy'] = np.mean(meanScores['Overall Accuracy'])
    meanScores['Raw Pitch Accuracy'] = np.mean(meanScores['Raw Pitch Accuracy'])
    meanScores['Raw Chroma Accuracy'] = np.mean(meanScores['Raw Chroma Accuracy'])
    meanScores['Voicing Recall'] = np.mean(meanScores['Voicing Recall'])
    meanScores['Voicing False Alarm'] = np.mean(meanScores['Voicing False Alarm'])
    print(meanScores)
    # save scores to data frame
    scores_path = os.path.join(outPath, '_all_scores.csv')
    score_summary_path = os.path.join(outPath, "_score_summary.csv")
    # WRITE MEAN SCORE TO FILE
    with open(score_summary_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in meanScores.items():
           writer.writerow([key, value])
   # WRITE ALL SCORES TO FILE
    with open(scores_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        line=['']
        for key, value in all_scores[0].items():
            line.append(key)
        writer.writerow(line)
        ind = 0
        for score in all_scores:
            line = []
            line.append(ind)
            for key, value in score.items():
                line.append(value)
            ind += 1
            writer.writerow(line)

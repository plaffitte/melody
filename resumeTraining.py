'''
This script defines a hybrid CNN-RNN model and trains it to extract the dominant melody from polyphonic music
Arguments to be given when calling script, in order of appearance:

            - Type of model: string (1D, 2D or BASELINE, VOICING, SOFTMAX)
                    - 1D: one dimensional targets/outputs and a recurrent layer to decode the sequence of   outputs in time.

                    - 2D: two dim targets/outputs, predictions of the entire time-frequency representation via convolutional layers.

                    - BASELINE: Rachel's model.

                    - VOICING/SOFTMAX: 1D model with softmax output activations to predict one frequency bin. Needs additional bin in time-frequency representation (targets and outputs) to store the voicing information.

            - Name of exp: string

            - File containing experimental parameters: string

Optional arguments:

            - Path to previously computed model: string

Example: > python main.py 1D test1 ~/test1model.h5
'''
import keras
import keras.backend as K
from keras.utils import plot_model
import glob, os, sys
import numpy as np
from data_creation import toyData, DataSet
from utils import log
import medleydb as mdb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from evaluation import calculate_metrics, plotScores, plotThreeScores, writeScores
from modelLearning import trainModel, trainStatefull, test, trainRnnFromCnn, testCNN, testDeepSalience, testStatefull
import mir_eval
from plotFunction import plot
import csv

###########################################################################################################
###########################################################################################################
### Algo params:
plotTargets = False
stateFull = True
###########################################################################################################
###########################################################################################################

if __name__ == "__main__":
    K.set_learning_phase(1)
    sys.stdout.flush()
    params = {}
    for i in range(1, (len(sys.argv)-1), 2):
        params[sys.argv[i].replace("--","")] = sys.argv[i+1]

    sys.stdout.flush()
    outPath = params['expDir']
    inputPath = os.path.join(outPath, 'features')
    targetPath = os.path.join(outPath, 'targets')
    testPath = os.path.join(outPath, 'test')
    trainPath = os.path.join(outPath, 'train')
    validPath = os.path.join(outPath, 'valid')
    locals().update(params)
    fftSize = int(nOctave) * int(binsPerOctave)
    modelDim = params['expName']
    voicing = True if params['voicing']=="True" or "SOFTMAX" in modelDim or "VOICING" in modelDim else False
    stateFull = params['stateFull']
    seqNumber = params['seqNumber']
    K.set_learning_phase(1)
    myModel = keras.models.load_model(params['model'])
    sys.stdout.flush()

    ### PREPARE DATA ###
    dataset = DataSet(inputPath, targetPath, modelDim)
    log('Formatting Training Dataset')
    with open(os.path.join(outPath, os.path.join('train', 'trainFileList.txt')), 'r') as f:
        trainSet = []
        for trainfile in f.readlines():
            trainSet.append(trainfile.replace('\n', ''))
    with open(os.path.join(outPath, os.path.join('test', 'testFileList.txt')), 'r') as f:
        testSet = []
        for testfile in f.readlines():
            testSet.append(testfile.replace('\n', ''))
    with open(os.path.join(outPath, os.path.join('valid', 'validFileList.txt')), 'r') as f:
        validSet = []
        for validFile in f.readlines():
            validSet.append(validFile.replace('\n', ''))

    ### TRAIN MODEL ###
    if stateFull:
        myModel, modelSplit = trainStatefull(
        params, dataset, trainSet, validSet, modelDim, outPath, voicing, fftSize, seqNumber)
    else:
        myModel, modelSplit = trainModel(
        params, dataset, trainSet, validSet, modelDim, outPath, voicing, fftSize)

    ### TEST MODEL ###
    if stateFull:
        preds, labs, realTestSet = testStatefull(train, myModel, modelSplit, dataset, testSet, params, modelDim, targetPath, inputPath, plotTargets, voicing, fftSize, seqNumber)
    else:
        preds, labs, realTestSet, cnnOut = test(train, myModel, modelSplit, dataset, testSet, params, modelDim, targetPath, plotTargets, voicing, fftSize, seqNumber)

    ### COMPUTE MIR METRICS ###
    if "SOFTMAX" in modelDim or "VOICING" in modelDim or "statefull" in modelDim:
        th = 0.5
    elif "TESTDEEPSALIENCE" in modelDim:
        th = 0.38
    else:
        th = 0.01

    if not cnnOnly and modelSplit:
        ### SCORES ON CNN PREDICTIONS
        all_scores, melodyEstimation = calculate_metrics(cnnOut, labs, testSet, int(binsPerOctave), int(nOctave), thresh=th, voicing=voicing)
        ### SCORES ON RNN PREDICTIONS
        all_scores, melodyEstimation = calculate_metrics(preds, labs, testSet, int(binsPerOctave), int(nOctave), thresh=th, voicing=voicing)
    else:
        print "SHAPES:", preds[0].shape
        all_scores, melodyEstimation = calculate_metrics(preds, labs, testSet, int(binsPerOctave), int(nOctave), thresh=th, voicing=voicing)

    ## SAVE (AND PLOT) TEST RESULTS ###
    p = []
    l = []
    c = []
    if len(preds[0].shape)==3:
        if "TESTDEEPSALIENCE" in modelDim:
            for (pred, lab) in zip(preds, labs):
                toto = np.zeros((1, int(fftSize)))
                toto2 = np.zeros((1, int(fftSize)))
                for i in range(pred.shape[0]):
                    toto = np.concatenate((toto, lab[i, :, :]), 0)
                    toto2 = np.concatenate((toto2, pred[i, :, :]), 0)
                    lim = np.min((toto.shape[-1], toto2.shape[-1])) # Cut to shortest's length
                labels = toto[:, 1:lim]
                predictions = toto2[:, 1:lim]
                p.append(predictions)
                l.append(labels)
        else:
            for (pred, lab, cnn) in zip(preds, labs, cnnOut):
                # pred = binarize(pred)
                if "SOFTMAX" in modelDim or "VOICING" in modelDim:
                    toto = np.zeros((int(fftSize)+1, 1))
                    toto2 = np.zeros((int(fftSize)+1, 1))
                    toto3 = np.zeros((int(fftSize)+1, 1))
                else:
                    toto = np.zeros((int(fftSize), 1))
                    toto2 = np.zeros((int(fftSize), 1))
                    toto3 = np.zeros((int(fftSize), 1))
                for i in range(pred.shape[0]):
                    for j in range(pred.shape[2]):
                        toto = np.concatenate((toto, lab[i, :, j, None]), 1)
                        toto2 = np.concatenate((toto2, pred[i, :, j, None]), 1)
                        toto3 = np.concatenate((toto3, cnn[i, :, j, None]), 1)
                        lim = np.min((toto.shape[-1], toto2.shape[-1])) # Cut to shortest's length
                labels = toto[:, 1:lim]
                predictions = toto2[:, 1:lim]
                cnnOutput = toto3[:, 1:lim]
                p.append(predictions)
                l.append(labels)
                c.append(cnnOutput)
        plotScores(outPath, p, l, realTestSet, testPath)
    else:
        if not modelSplit:
            plotScores(outPath, preds, labs, realTestSet, testPath)
        else:
            plotThreeScores(outPath, preds, labs, cnnOut, realTestSet, testPath)

    ### WRITE SCORES TO FILE
    writeScores(all_scores, outPath)

    testFile = glob.glob(os.path.join(targetPath, '{}_mel1_output.npy'.format(testSet[0])))
    testArray = np.load(testFile[0])
    lim = np.min((labs[0].T.shape[1], testArray.shape[1]))
    print labs[0].shape[1], testArray.shape[0]
    if testArray.shape[0] == (labs[0].shape[1] - 1):
        diff = labs[0].T[1:, :lim] - testArray[:, :lim]
    elif testArray.shape[0] == labs[0].shape[1]:
        diff = labs[0].T[:, :lim] - testArray[:, :lim]
    plt.plot(diff)
    plt.show()

    plot(outPath)
    del myModel

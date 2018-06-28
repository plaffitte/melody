'''
This script performs Dominant Melody Transcription as per the 'melody2' definition in
http://medleydb.weebly.com/description.html
Arguments to be given when calling script, in order of appearance:
            - Type of model: string (1D, 2D or BASELINE, VOICING, SOFTMAX)
                    - 1D: one dimensional targets/outputs and a recurrent layer to decode the sequence of   outputs in time.

                    - 2D: two dim targets/outputs, predictions of the entire time-frequency representation via convolutional layers.

                    - BASELINE: Deep Salience model (Bittner et. al. ISMIR 2017).

                    - VOICING/SOFTMAX: 1D model with softmax output activations to predict one frequency bin. Needs additional bin in time-frequency representation (targets and outputs) to store the voicing information.
            - Name of exp: string

Optional arguments:
            - Path to previously computed model: string
Example: > python main.py 1D test ~/testModel.h5
'''
import keras
import glob, os, sys
import numpy as np
from data_creation import toyData, DataSet
from utils import log
import medleydb as mdb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from evaluation import calculate_metrics, plotScores, plotThreeScores
from modelLearning import trainModel, test, trainRnnFromCnn, testCNN
import mir_eval
from plotFunction import plot
import csv

### Algo params:
generateData = True
plotTargets = False
trainRNNonly = False
cnnOnly= False
##########################################################################
if __name__ == "__main__":

    params = {}
    for i in range(1, (len(sys.argv)-1), 2):
        params[sys.argv[i].replace("--","")] = sys.argv[i+1]

    outPath = params['expDir']
    if os.path.exists(os.path.join(outPath, 'features')) and os.path.exists(os.path.join(outPath, 'targets')):
        generateData = False
        inputPath = os.path.join(outPath, 'features')
        targetPath = os.path.join(outPath, 'targets')
    else:
        inputPath = '/data/scratch/rmb456/multif0_ismir2017/training_data_with_blur/melody1/inputs'
        targetPath = '/data/scratch/rmb456/multif0_ismir2017/training_data_with_blur/melody1/outputs'
    globals().update(params)
    fftSize = int(nOctave) * int(binsPerOctave)
    modelDim = params['expName']
    voicing = True if params['voicing']=="True" or "SOFTMAX" in modelDim else False
    if 'model' in params:
        modelSplit = False
        train = False
        # cnnOnly = True
        myModel = keras.models.load_model(params['model'])
    else:
        train = True
    if "TESTDEEPSALIENCE" in modelDim: ### TEST DEEP SALIENCE MODEL DIRECTLY ON MEDLEYDB DATABASE
        train = False
    testPath = os.path.join(outPath, 'test')
    trainPath = os.path.join(outPath, 'train')
    validPath = os.path.join(outPath, 'valid')
    if not os.path.isdir(outPath):
        os.mkdir(outPath)
    if not os.path.isdir(trainPath):
        os.mkdir(trainPath)
    if not os.path.isdir(testPath):
        os.mkdir(testPath)
    if not os.path.isdir(validPath):
        os.mkdir(validPath)

    if "TESTDEEPSALIENCE" in modelDim:
        dataset = DataSet(inputPath, targetPath, "rachel")
    else:
        dataset = DataSet(inputPath, targetPath, modelDim)

    ### PREPARE DATA ###
    log('Formatting Training Dataset')
    if os.path.exists(trainPath) and os.path.exists(os.path.join(trainPath, 'trainFileList.txt')):
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
    else:
        trainSet, validSet, testSet = dataset.partDataset()
        with open(os.path.join(trainPath, 'trainFileList.txt'), 'w') as f:
            for el in trainSet:
                f.write(el+'\n')
        with open(os.path.join(testPath, 'testFileList.txt'), 'w') as f:
            for el in testSet:
                f.write(el+'\n')
        with open(os.path.join(validPath, 'validFileList.txt'), 'w') as f:
            for el in validSet:
                f.write(el+'\n')

    if generateData:
        dataList = trainSet + validSet + testSet
        dataset.getFeature(dataList, modelDim, outPath, int(binsPerOctave), int(nOctave), nHarmonics=int(nHarmonics), homemade=False)

    ### TRAIN VOICING MODEL FIRST ###
    if train:
        myModel, modelSplit = trainModel(
        params, dataset, trainSet, validSet, "VOICING", outPath, voicing, fftSize
        )

    ### TEST MODEL ###
    if not train:
        testSet = []
        with open(os.path.join(testPath, 'testFileList.txt'), 'r') as f:
            testSet = f.read().splitlines()
    if cnnOnly:
        preds, labs, realTestSet = testCNN(train, myModel, dataset, testSet, params, modelDim, targetPath, plotTargets, voicing,fftSize)
    else:
        if "TESTDEEPSALIENCE" in modelDim:
            preds, labs, realTestSet = testDeepSalience(dataset, testSet, params, "rachel", targetPath, fftSize)
        else:
            preds, labs, realTestSet, cnnOut = test(train, myModel, modelSplit, dataset, testSet, params, modelDim, targetPath, plotTargets, voicing, fftSize)

    ### COMPUTE MIR METRICS ###
    if "SOFTMAX" in modelDim:
        th = 0.5
    else:
        th = 0.05

    if not cnnOnly and modelSplit:
        ### SCORES ON CNN PREDICTIONS
        all_scores, melodyEstimation = calculate_metrics(cnnOut, labs, testSet, int(binsPerOctave), int(nOctave), thresh=th, voicing=voicing)
        ### SCORES ON RNN PREDICTIONS
        all_scores, melodyEstimation = calculate_metrics(preds, labs, testSet, int(binsPerOctave), int(nOctave), thresh=th, voicing=voicing)
    else:
        all_scores, melodyEstimation = calculate_metrics(preds, labs, testSet, int(binsPerOctave), int(nOctave), thresh=th, voicing=voicing)
    ### SAVE (AND PLOT) TEST SCORES ###
    p = []
    l = []
    c = []
    if len(preds[0].shape)==3:
        for (pred, lab, cnn) in zip(preds, labs, cnnOut):
            # pred = binarize(pred)
            if "SOFTMAX" in modelDim or "1D" in modelDim:
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
        plotScores(outPath, p, l, c, all_scores, realTestSet, testPath)
    else:
        if not modelSplit:
            plotScores(outPath, preds, labs, all_scores, realTestSet, testPath)
        else:
            plotThreeScores(outPath, preds, labs, cnnOut, all_scores, realTestSet, testPath)

    plot(outPath)
    del myModel

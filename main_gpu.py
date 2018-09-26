'''
This script defines a hybrid CNN-RNN model and trains it to extract the dominant melody from polyphonic music
Arguments to be given when calling script, in order of appearance:

            - Type of model: string (1D, 2D or BASELINE, VOICING, SOFTMAX)
                    - 1D: one dimensional targets/outputs and a recurrent layer to decode the sequence of   outputs in time.

                    - 2D: two dim targets/outputs, predictions of the entire time-frequency representation via convolutional layers.

                    - BASELINE: RNN on top of Rachel's deep salience model.

                    - VOICING/SOFTMAX: 1D model with softmax output activations to predict one frequency bin. Needs additional bin in time-frequency representation (targets and outputs) to store the voicing information.

            - Name of exp: string

            - File containing experimental parameters: string

Optional arguments:

            - Path to previously computed model: string

Example: > python main.py 1D test1 ~/test1model.h5
'''
import traceback
import keras
import keras.backend as K
from keras.utils import plot_model
import glob, os, sys, re
import numpy as np
from data_creation import DataSet, getLabels
from utils import log
import medleydb as mdb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from evaluation import calculate_metrics, writeScores, plotLinear
from modelLearning import trainModel, test, testCNN, testDeepSalience
import mir_eval
from plotFunction import plot
import csv
import manage_gpus as gpl
from tensorflow.python.client import device_lib
import time
try:
    import cPickle as pickle
except:
    import pickle
########################### GPU HANDLING CODE ###########################
try:
    GPU = True
    gpu_ids = gpl.board_ids()
    if gpu_ids is None:
         # system does not have a GPU so don't bother locking one, directly use CPU
         gpu_device=None
    else:
         # select any free gpu
         gpu_device=-1

    gpu_id_locked = -1
    # gpu_device = None
    if gpu_device is not None:
        gpu_id_locked = gpl.obtain_lock_id(id=gpu_device)
        if gpu_id_locked < 0:
            # automatic lock removal has time delay of 2 so be sure to have the lock of the last run removed we wait
            # for 3 s here
            time.sleep(3)
            gpu_id_locked=gpl.obtain_lock_id(id=args.gpu_device)
            if gpu_id_locked < 0:
                if args.gpu_device < 0:
                    raise RuntimeError("No GPUs available for locking")
                else:
                    raise RuntimeError("cannot obtain any of the selected GPUs {0}".format(str(args.gpu_device)))

        # obtain_lock_id positions CUDA_VISIBLE_DEVICES such that only the selected GPU is visibale,
        # therefore we need now select /GPU:0
        comp_device = "/GPU:0"
    else:
        comp_device = "/cpu:0"
        os.environ["CUDA_VISIBLE_DEVICES"]=""
    print(device_lib.list_local_devices())
except:
    pass
###########################################################################################################
###########################################################################################################
### Algo params:
generateData = True
plotTargets = False
trainRNNonly = False
cnnOnly = False
if os.path.isdir('/data2/anasynth_nonbp/laffitte'):
    audioPath = '/data2/anasynth_nonbp/laffitte/MedleyDB/Audio/'
    annotPath = '/data2/anasynth_nonbp/laffitte/MedleyDB/Annotations/Melody_Annotations/MELODY2'
else:
    annotPath = '/net/as-sdb/data/mir2/MedleyDB/Annotations/Melody_Annotations/MELODY2'
    audioPath = '/net/as-sdb/data/mir2/MedleyDB/Audio/'
###########################################################################################################
###########################################################################################################

if __name__ == "__main__":
    try:
        # K.set_learning_phase(1)
        sys.stdout.flush()
        params = {}
        for i in range(1, (len(sys.argv)-1), 2):
            params[sys.argv[i].replace("--","")] = sys.argv[i+1]
        sys.stdout.flush()
        locals().update(params)
        fftSize = int(nOctave)*int(binsPerOctave)
        modelDim = params['expName']
        voicing = True if params['voicing']=="True" or "SOFTMAX" in modelDim or "VOICING" in modelDim else False
        stateFull = bool(params['stateFull'])
        seqNumber = int(params['seqNumber'])
        outPath = params['expDir']
        batchSize = int(params['batchSize'])
        nUnits = int(params['nUnits'])
        inputPath = os.path.join(outPath, 'features')
        targetPath = os.path.join(outPath, 'targets')
        log("MODELDIM IS:", modelDim)
        if "TESTDEEPSALIENCE" in modelDim:
            dataobj = DataSet(inputPath, targetPath, "rachel")
        else:
            dataobj = DataSet(inputPath, targetPath, modelDim)
        if not os.path.exists(outPath):
            os.mkdir(outPath)
        if os.path.exists(inputPath) and os.path.exists(targetPath):
            generateData = False
        else:
            os.mkdir(os.path.join(outPath, 'features'))
            os.mkdir(os.path.join(outPath, 'targets'))

        if 'model' in params:
            modelSplit = False
            train = False
            # cnnOnly = True
            myModel = keras.models.load_model(params['model'])
            sys.stdout.flush()
        elif 'modelCnn' in params and 'modelRnn' in params:
            modelSplit = True
            train = False
            myModel = [keras.models.load_model(params['modelCnn']), keras.models.load_model(params['modelRnn'])]
        elif 'modelCnn' in params:
            modelSplit = True
            convModel = keras.models.load_model(params['modelCnn'])
            trainRNNonly = True
            train = False
        else:
            modelSplit = False
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

        ### PREPARE DATA ###
        log('Formatting Dataset')
        if train:
            if os.path.exists(trainPath) and os.path.exists(os.path.join(trainPath, 'trainFileList.txt')):
                with open(os.path.join(outPath, os.path.join('train', 'trainFileList.txt')), 'r') as f:
                    log('...training data')
                    trainSet = []
                    for trainfile in f.readlines():
                        trainSet.append(trainfile.replace('\n', ''))
                with open(os.path.join(outPath, os.path.join('test', 'testFileList.txt')), 'r') as f:
                    log('...validation data')
                    testSet = []
                    for testfile in f.readlines():
                        testSet.append(testfile.replace('\n', ''))
                with open(os.path.join(outPath, os.path.join('valid', 'validFileList.txt')), 'r') as f:
                    log('...test data')
                    validSet = []
                    for validFile in f.readlines():
                        validSet.append(validFile.replace('\n', ''))
            else:
                trainSet, validSet, testSet = dataobj.partDataset()
                with open(os.path.join(trainPath, 'trainFileList.txt'), 'w') as f:
                    for el in trainSet:
                        f.write(el+'\n')
                with open(os.path.join(testPath, 'testFileList.txt'), 'w') as f:
                    for el in testSet:
                        f.write(el+'\n')
                with open(os.path.join(validPath, 'validFileList.txt'), 'w') as f:
                    for el in validSet:
                        f.write(el+'\n')
        else:
            testSet = []
            trainSet = []
            validSet = []
            with open(os.path.join(testPath, 'testFileList.txt'), 'r') as f:
                testSet = f.read().splitlines()

        if generateData:
            dataList = trainSet + validSet + testSet
            dataobj.getFeature(dataList, modelDim, outPath, int(binsPerOctave), int(nOctave), int(nHarmonics), False)
        realTestSet = []
        realTrainSet = []
        realValidSet = []
        dirList = sorted(os.listdir(audioPath))
        dirList = [i for i in dirList if '.' not in i]
        for j in dirList:
            fileList = sorted(os.listdir(os.path.join(audioPath, j)))
            fileList = [k[:-8] for k in fileList if re.match('[^._].*?.wav', k)]
            trainTracks = [tr for tr in fileList if tr in trainSet]
            for i in trainTracks:
                annotFile = os.path.join(annotPath, i+'_MELODY2.csv')
                if annotFile is not None and os.path.exists(annotFile):
                    realTrainSet.extend(trainTracks)
            validTracks = [tr for tr in fileList if tr in validSet]
            for i in validTracks:
                annotFile = os.path.join(annotPath, i+'_MELODY2.csv')
                if annotFile is not None and os.path.exists(annotFile):
                    realValidSet.extend(validTracks)
            testTracks = [tr for tr in fileList if tr in testSet]
            for i in testTracks:
                annotFile = os.path.join(annotPath, i+'_MELODY2.csv')
                if annotFile is not None and os.path.exists(annotFile):
                    realTestSet.extend(testTracks)
        ### TRAIN MODEL ###
        if train:
            myModel, modelSplit = trainModel(
            params, dataobj, realTrainSet, realValidSet, modelDim, outPath, nUnits, voicing, fftSize, seqNumber
            )
        if trainRNNonly:
            myModel, modelSplit = trainRnnFromCnn(
            convModel, params, dataobj, realTrainSet, realValidSet, modelDim, outPath, voicing, fftSize
            )

        ### TEST MODEL ###
        if cnnOnly:
            preds, labs, realTestSet = testCNN(train, myModel, dataobj, realTestSet, params, modelDim, targetPath, plotTargets, voicing, fftSize)
        else:
            if "TESTDEEPSALIENCE" in modelDim:
                preds, labs, realTestSet, cnnOut = testDeepSalience(dataobj, realTestSet, params, "rachel", targetPath, fftSize)
            else:
                if os.path.exists(os.path.join(outPath, "outputs")):
                    preds = np.load(os.path.join(os.path.join(outPath, "outputs"), "outputs.npy"))
                    labs = np.load(os.path.join(os.path.join(outPath, "labels"), "labels.npy"))
                    inputs = np.load(os.path.join(os.path.join(outPath, "inputs"), "inputs.npy"))
                    with open(os.path.join(outPath, 'trackList.txt'), 'r') as f:
                        trackList = []
                        for song in f.readlines():
                            trackList.append(song.replace('\n', ''))
                else:
                    log("Testing Model!")
                    preds, labs, inputs, trackList = test(train, myModel, dataobj, realTestSet, params, modelDim, targetPath, voicing, fftSize, seqNumber)
                    os.mkdir(os.path.join(outPath, "outputs"))
                    os.mkdir(os.path.join(outPath, "labels"))
                    os.mkdir(os.path.join(outPath, "inputs"))
                    with open(os.path.join(os.path.join(outPath, "outputs"), "outputs.npy"), 'wb') as outfile:
                        pickle.dump(preds, outfile, pickle.HIGHEST_PROTOCOL)
                    with open(os.path.join(os.path.join(outPath, "labels"), "labels.npy"), 'wb') as outfile:
                        pickle.dump(labs, outfile, pickle.HIGHEST_PROTOCOL)
                    with open(os.path.join(os.path.join(outPath, "inputs"), "inputs.npy"), 'wb') as outfile:
                        pickle.dump(inputs, outfile, pickle.HIGHEST_PROTOCOL)
                    with open(os.path.join(outPath, 'trackList.txt'), 'w') as f:
                        for song in trackList:
                            f.write(song+'\n')

        if "SOFTMAX" in modelDim or "CATEGORICAL" in modelDim:
            th = 0.5
        elif "TESTDEEPSALIENCE" in modelDim:
            th = 0.38
        elif "BASELINE" in modelDim:
            th = 0.1
        else:
            th = 0.01

        ## COMPUTE SCORES on MODEL'S predictions
        if not cnnOnly and modelSplit:
            all_scores, melodyEstimation, refMelody, inputs = calculate_metrics(dataobj, cnnOut, labs, inputs, trackList, int(binsPerOctave), int(nOctave), testPath, batchSize, seqNumber, thresh=th, voicing=voicing)
            all_scores, melodyEstimation, refMelody, inputs = calculate_metrics(dataobj, preds, labs, inputs, trackList, int(binsPerOctave), int(nOctave), testPath, batchSize, seqNumber, thresh=th, voicing=voicing)
        else:
            all_scores, melodyEstimation, refMelody, inputs = calculate_metrics(dataobj, preds, labs, inputs, trackList, int(binsPerOctave), int(nOctave), testPath, batchSize, seqNumber, thresh=th, voicing=voicing)

        ## Write SCORES to CSV file and PDF graph ###
        writeScores(all_scores, outPath)
        plot(outPath)

        ### COMPUTE SCORES for DEEP SALIENCE MODEL representations to use as baseline comparison ###
        if "BASELINE" in modelDim:
            deepSaliencePath = outPath+'deepSalienceInputs'
            if not os.path.isdir(deepSaliencePath):
                os.mkdir(deepSaliencePath)
            log('Computing scores for Deep Salience Model')
            ## GET LABELS ###
            if os.path.exists(os.path.join(deepSaliencePath, "labels")):
                labs = np.load(os.path.join(os.path.join(deepSaliencePath, "labels"), "labels.npy"))
                inputs = np.load(os.path.join(os.path.join(deepSaliencePath, "inputs"), "inputs.npy"))
                with open(os.path.join(deepSaliencePath, 'trackList.txt'), 'r') as f:
                    trackList = []
                    for song in f.readlines():
                        trackList.append(song.replace('\n', ''))
            else:
                labs, inputs, trackList = getLabels(dataobj, realTestSet, params, modelDim, False, fftSize, seqNumber)
                os.mkdir(os.path.join(deepSaliencePath, "labels"))
                os.mkdir(os.path.join(deepSaliencePath, "inputs"))
                with open(os.path.join(os.path.join(deepSaliencePath, "labels"), "labels.npy"), 'wb') as outfile:
                    pickle.dump(labs, outfile, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(os.path.join(deepSaliencePath, "inputs"), "inputs.npy"), 'wb') as outfile:
                    pickle.dump(inputs, outfile, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(deepSaliencePath, 'trackList.txt'), 'w') as f:
                    for song in trackList:
                        f.write(song+'\n')
            all_scores, melodyEstimation, _, _ = calculate_metrics(dataobj, inputs, labs, inputs, trackList, int(binsPerOctave), int(nOctave), deepSaliencePath, batchSize, seqNumber, thresh=0.05, voicing=False)
            writeScores(all_scores, deepSaliencePath)
        del myModel
    except Exception as err:
        print(sys.exc_info()[0])
        print(err)
        traceback.print_tb(err.__traceback__)
if GPU:
    if gpu_id_locked >= 0:
        gpl.free_lock(gpu_id_locked)

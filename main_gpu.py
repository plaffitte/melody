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
from evaluation import calculate_metrics, arrangePredictions, writeScores
from modelLearning import trainModel, trainStatefull, test, testCNN, testDeepSalience, testStatefull
import mir_eval
from plotFunction import plot
import csv
import manage_gpus as gpl
from tensorflow.python.client import device_lib
import time
########################### GPU HANDLING CODE ###########################
gpu_ids = gpl.board_ids()
if gpu_ids is None:
     # system does not have a GPU so don't bother locking one, directly use CPU
     gpu_device=None
else:
     # select any free gpu
     gpu_device=-1

gpu_id_locked = -1
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
###########################################################################################################
###########################################################################################################
### Algo params:
generateData = True
plotTargets = False
trainRNNonly = False
cnnOnly = False
###########################################################################################################
###########################################################################################################

if __name__ == "__main__":
    K.set_learning_phase(1)
    sys.stdout.flush()
    params = {}
    for i in range(1, (len(sys.argv)-1), 2):
        params[sys.argv[i].replace("--","")] = sys.argv[i+1]

    sys.stdout.flush()
    locals().update(params)
    fftSize = int(nOctave) * int(binsPerOctave)
    modelDim = params['expName']
    voicing = True if params['voicing']=="True" or "SOFTMAX" in modelDim or "VOICING" in modelDim else False
    stateFull = params['stateFull']
    seqNumber = int(params['seqNumber'])
    outPath = params['expDir']
    inputPath = os.path.join(outPath, 'features')
    targetPath = os.path.join(outPath, 'targets')
    log("MODELDIM IS:", modelDim)
    if "TESTDEEPSALIENCE" in modelDim:
        dataset = DataSet(inputPath, targetPath, "rachel")
    elif "BASELINE" in modelDim:
        path = "/data/anasynth_nonbp/laffitte/Experiments/deepSalienceRepresentations"
        inputPath = os.path.join(path, 'features')
        targetPath = os.path.join(path, 'targets')
        dataset = DataSet(inputPath, targetPath, modelDim)
    else:
        dataset = DataSet(inputPath, targetPath, modelDim)
    if os.path.exists(inputPath) and os.path.exists(targetPath):
        generateData = False
    else:
        os.mkdir(os.path.join(outPath, 'features'))
        os.mkdir(os.path.join(outPath, 'targets'))

    if 'model' in params:
        modelSplit = False
        train = False
        # cnnOnly = True
        K.set_learning_phase(1)
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

    ### TRAIN MODEL ###
    if train:
        if stateFull:
            myModel, modelSplit = trainStatefull(
            params, dataset, trainSet, validSet, modelDim, outPath, voicing, fftSize, seqNumber)
        else:
            myModel, modelSplit = trainModel(
            params, dataset, trainSet, validSet, modelDim, outPath, voicing, fftSize
            )
    if trainRNNonly:
        myModel, modelSplit = trainRnnFromCnn(
        convModel, params, dataset, trainSet, validSet, modelDim, outPath, voicing, fftSize
        )

    ### TEST MODEL ###
    if not train:
        testSet = []
        with open(os.path.join(testPath, 'testFileList.txt'), 'r') as f:
            testSet = f.read().splitlines()
    if cnnOnly:
        preds, labs, realTestSet = testCNN(train, myModel, dataset, testSet, params, modelDim, targetPath, plotTargets, voicing, fftSize)
    else:
        if "TESTDEEPSALIENCE" in modelDim:
            preds, labs, realTestSet, cnnOut = testDeepSalience(dataset, testSet, params, "rachel", targetPath, fftSize)
        elif stateFull:
            preds, labs, inputs, realTestSet = testStatefull(train, myModel, modelSplit, dataset, testSet, outPath, params, modelDim, targetPath, inputPath, plotTargets, voicing, fftSize, seqNumber)
        else:
            preds, labs, realTestSet, cnnOut = test(train, myModel, modelSplit, dataset, testSet, params, modelDim, targetPath, plotTargets, voicing, fftSize, seqNumber)

    if "SOFTMAX" in modelDim or "CATEGORICAL" in modelDim:
        th = 0.5
    elif "TESTDEEPSALIENCE" in modelDim:
        th = 0.38
    elif "BASELINE" in modelDim:
        th = 0.1
    else:
        th = 0.01

    ### COMPUTE SCORES on MODEL'S predictions
    if not cnnOnly and modelSplit:
        all_scores, melodyEstimation = calculate_metrics(cnnOut, labs, testSet, int(binsPerOctave), int(nOctave), thresh=th, voicing=voicing)
        all_scores, melodyEstimation = calculate_metrics(preds, labs, testSet, int(binsPerOctave), int(nOctave), thresh=th, voicing=voicing)
    else:
        all_scores, melodyEstimation = calculate_metrics(preds, labs, testSet, int(binsPerOctave), int(nOctave), thresh=th, voicing=voicing)
    writeScores(all_scores, outPath)
    plot(outPath)

    if "BASELINE" in modelDim:
        ## COMPUTE SCORES for DEEP SALIENCE MODEL representations
        fileList = os.listdir(inputPath)
        fileList = [tr[:-15] for tr in fileList if tr[:-15] in testSet]
        dsRep = []
        labels = []
        for f in fileList:
            pred = np.load(os.path.join(inputPath, '{}_mel1_input.npy'.format(f)))
            lab = np.load(os.path.join(targetPath, '{}_mel1_output.npy'.format(f)))
            dsRep.append(pred)
            labels.append(lab)
        deepSaliencePath = outPath+'deepSalienceInputs'
        if not os.path.isdir(deepSaliencePath):
            os.mkdir(deepSaliencePath)
        log('Computing scores for Deep Salience Model')
        all_scores, melodyEstimation = calculate_metrics(dsRep, labels, testSet, int(binsPerOctave), int(nOctave), thresh=0.38, voicing=voicing)
        writeScores(all_scores, deepSaliencePath)

    ### WRITE and PLOT SCORES
    # arrangePredictions(dsRep, labels, inputs, testSet, deepSaliencePath)
    print("SHAPE OF PREDICTIONS", len(preds, preds[0].shape))
    arrangePredictions(preds, labs, inputs, testSet, testPath)

    # testFile = glob.glob(os.path.join(targetPath, '{}_mel1_output.npy'.format(testSet[0])))
    # testArray = np.load(testFile[0])
    # lim = np.min((labs[0].T.shape[1], testArray.shape[1]))
    # if testArray.shape[0] == (labs[0].shape[1] - 1):
    #     diff = labs[0].T[1:, :lim] - testArray[:, :lim]
    # elif testArray.shape[0] == labs[0].shape[1]:
    #     diff = labs[0].T[:, :lim] - testArray[:, :lim]
    # plt.plot(diff)
    # plt.show()

    del myModel

### Release GPU
if gpu_id_locked >= 0:
    gpl.free_lock(gpu_id_locked)

import numpy as np
import os, glob
import sys
from predict_on_audio import load_model
from data_creation import toyData, DataSet
from modelLearning import trainStatefull, testStatefull, trainModel, test
from utils import log, binarize
from evaluation import calculate_metrics, plotScores, plotThreeScores, writeScores
import matplotlib.pyplot as plt
from model import model
from plotFunction import plot
import keras
import manage_gpus as gpl
from tensorflow.python.client import device_lib
import time

RUNDEEPSALIENCE = False
TRAIN = False
TEST = True

# Initialize some variables
rnnBatch = 16
timeDepth = 50
nHarmonics = 6
batchSize = 1000 # 1s of signal with 11ms hopsize (which is the case for medleyDB dataset)
binsPerOctave = 60
nOctave = 6
voicing = True
fftSize = binsPerOctave*nOctave

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

def savePlot(preds, labs, realTestSet, testPath):
    ## SAVE (AND PLOT) TEST RESULTS ###
    p = []
    l = []
    print(preds[0].shape)
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

def predictDeepSalience(path, pathOut, dataSet):
    # Load data
    pathLab = os.path.join(path, 'targets')
    pathFeat = os.path.join(path, 'features')
    fileList = os.listdir(pathLab)
    fileList = [tr[:-16] for tr in fileList if tr[:-16] in dataSet]
    predictions = []
    labels = []
    for f in fileList:
        log("Processing track: {}".format(f))
        curInput = np.load(os.path.join(pathFeat, '{}_mel1_input.npy'.format(f)))
        curTarget = np.load(os.path.join(pathLab, '{}_mel1_output.npy'.format(f)))
        curInput = curInput.transpose(1, 2, 0)[None,:,:,:]
        n_t = curInput.shape[2]
        # Create batches
        output = []
        length = curTarget.shape[-1]
        n_slices = 2000
        t_slices = list(np.arange(0, n_t, n_slices))
        for i, t in enumerate(t_slices):
            dsRep = deepModel.predict(curInput[:, :, t:t+n_slices, :])
            output.append(dsRep[0,:,:])
        output = np.hstack(output)
        np.save(os.path.join(pathOut, '{}_mel1_input.npy'.format(f)), output.astype(np.float32))
        # output = binarize(output)
        predictions.append(output)
        labels.append(curTarget)

    return predictions, labels

# Set paths
path = '/data/anasynth_nonbp/laffitte/Experiments/inputs'
outPath = '/data/anasynth_nonbp/laffitte/Experiments/deepSalienceRepresentations'
if not os.path.isdir(outPath):
    os.mkdir(outPath)
inputPath = os.path.join(outPath, 'features')
targetPath = os.path.join(outPath, 'targets')
dataset = DataSet(inputPath, targetPath, "rachel")
trainSet, validSet, testSet = dataset.partDataset()
dataList = trainSet + validSet + testSet
dataset.getFeature(dataList, "rachel", outPath, int(60), int(6), nHarmonics=int(6), homemade=False)
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

####################################### RUN DEEP SALIENCE MODEL #############################################
pathFigDsr = os.path.join(testPath, 'deepSalienceRepresentationGraphs')
if not os.path.isdir(pathFigDsr):
    os.mkdir(pathFigDsr)
if RUNDEEPSALIENCE:
    # Load and initialize deep salience model
    deepModel = load_model('melody2')
    print(deepModel.summary())
    _ = deepModel.predict(np.zeros((2,360,500,6))) # Initialize model
    # _,_ = predictDeepSalience(path, inputPath, trainSet)
    # _,_ = predictDeepSalience(path, inputPath, validSet)
    predictions, labels = predictDeepSalience(path, inputPath, testSet)
    savePlot(predictions, labels, testSet, pathFigDsr)
else:
    # Grab already computed deep salience representation data and compute scores on it
    fileList = os.listdir(targetPath)
    fileList = [tr[:-16] for tr in fileList if tr[:-16] in testSet]
    predictions = []
    labels = []
    for f in fileList:
        pred = np.load(os.path.join(inputPath, '{}_mel1_input.npy'.format(f)))
        lab = np.load(os.path.join(targetPath, '{}_mel1_output.npy'.format(f)))
        predictions.append(pred)
        labels.append(lab)
## COMPUTE SCORES FOR DEEP SALIENCE
log('Computing scores for Deep Salience Model')
all_scores, melodyEstimation = calculate_metrics(predictions, labels, testSet, binsPerOctave, nOctave, thresh=0.05, voicing=voicing)
writeScores(all_scores, pathFigDsr)

########################################### ADD RNN #######################################################
if TRAIN:
    modelDim = "deepsalience-BASELINE"
    dataset = DataSet(inputPath, targetPath, modelDim)
    log('Building temporal models with RNN on Deep Salience predictions')
    params = {}
    params['batchSize'] = batchSize
    params['timeDepth'] = timeDepth
    params['nHarmonics'] = nHarmonics
    params['hopSize'] = timeDepth
    params['nEpochs'] = 100
    params['fftSize'] = fftSize
    params['stateFull'] = True
    ### TRAIN RNN ON DEEP SALIENCE REPRESENTATION
    myModel, modelSplit = trainStatefull(
    params, dataset, trainSet, validSet, modelDim, outPath, voicing, fftSize, rnnBatch)
    # myModel, modelSplit = trainModel(
    # params, dataset, trainSet, validSet, modelDim, outPath, voicing, fftSize
    # )

if TEST:
    ### TEST RNN ON DEEP SALIENCE REPRESENTATION
    preds, labs, realTestSet = testStatefull(True, myModel, False, dataset, testSet, params, modelDim, targetPath, inputPath, False, voicing, fftSize, 50, True)
    # preds, labs, realTestSet = test(True, myModel, False, dataset, testSet, params, modelDim, targetPath, False, voicing, fftSize, 50, True)

    ### CALCULATE SCORES
    all_scores, melodyEstimation = calculate_metrics(preds, labs, testSet, int(binsPerOctave), int(nOctave), thresh=0.15, voicing=voicing)
    savePlot(preds, labs,realTestSet, testPath)
    writeScores(all_scores, outPath)
    plot(outPath)
    del myModel

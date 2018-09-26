import numpy as np
import os, glob
import sys
print("--> Using python version: ", sys.version)
from predict_on_audio import load_model
from data_creation import DataSet, getLabels
from modelLearning import trainModel, test
from utils import log, binarize
from evaluation import calculate_metrics, plotScores, plotThreeScores, writeScores
import matplotlib.pyplot as plt
from model import model
from plotFunction import plot
import keras
import manage_gpus as gpl
from tensorflow.python.client import device_lib
import time

GENERATEDATA = True if sys.argv[1]=="True" else False
RUNDEEPSALIENCE = True if sys.argv[2]=="True" else False
TRAIN = True if sys.argv[3]=="True" else False
TEST = True if sys.argv[4]=="True" else False


# Initialize some variables
rnnBatch = 16
timeDepth = 50
nHarmonics = 6
batchSize = 1000 # 1s of signal with 11ms hopsize (which is the case for medleyDB dataObj)
binsPerOctave = 60
nOctave = 6
voicing = False
fftSize = binsPerOctave*nOctave
params = {}
modelDim = "BASELINE"
batchSize = 1000
stateFull = "False"
Fs = 22050
params['batchSize'] = batchSize
params['timeDepth'] = timeDepth
params['nHarmonics'] = nHarmonics
params['hopSize'] = timeDepth
params['nEpochs'] = 100
params['fftSize'] = fftSize
params['stateFull'] = True
########################### GPU HANDLING CODE ###########################
try:
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
except:
    pass

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
    pathLab = os.path.join(path, 'targets')
    pathFeat = os.path.join(path, 'features')
    fileList = os.listdir(pathLab)
    fileList = [tr[:-16] for tr in fileList if tr[:-16] in dataSet]
    predictions = []
    for f in fileList:
        log("Processing track: {}".format(f))
        curInput = np.load(os.path.join(pathFeat, '{}_mel2_input.npy'.format(f)))
        curTarget = np.load(os.path.join(pathLab, '{}_mel2_target.npy'.format(f)))
        if len(curInput.shape)==3:
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
            np.save(os.path.join(pathOut, '{}_mel2_input.npy'.format(f)), output.astype(np.float32))
            # output = binarize(output)
            predictions.append(output)

    return predictions

#############################################################################################################
# Set paths
# outPath = '/data/anasynth_nonbp/laffitte/Experiments/deepSalienceRepresentations/outputs'
# path = '/data/anasynth_nonbp/laffitte/Experiments/deepSalienceRepresentations'
# if not os.path.isdir(outPath):
#     os.mkdir(outPath)
# inputPath = os.path.join(path, 'features')
# targetPath = os.path.join(path, 'targets')
# dataObj = DataSet(inputPath, targetPath, "BASELINE")
outPath = '/u/anasynth/laffitte/test/outputs'
path = '/u/anasynth/laffitte/test'
if not os.path.isdir(outPath):
    os.mkdir(outPath)
inputPath = os.path.join(path, 'features')
targetPath = os.path.join(path, 'targets')
dataObj = DataSet(inputPath, targetPath, modelDim)

if GENERATEDATA:
    trainSet, validSet, testSet = dataObj.partDataset()
    dataList = trainSet + validSet + testSet
    dataObj.getFeature(dataList, "rachel", path, 60, 6, nHarmonics=6, homemade=False)
testPath = os.path.join(path, 'test')
trainPath = os.path.join(path, 'train')
validPath = os.path.join(path, 'valid')
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
    with open(os.path.join(trainPath, 'trainFileList.txt'), 'r') as f:
        trainSet = []
        log('Reading training files list')
        for trainfile in f.readlines():
            trainSet.append(trainfile.replace('\n', ''))
    with open(os.path.join(testPath, 'testFileList.txt'), 'r') as f:
        testSet = []
        log('Reading test files list')
        for testfile in f.readlines():
            testSet.append(testfile.replace('\n', ''))
    with open(os.path.join(validPath, 'validFileList.txt'), 'r') as f:
        validSet = []
        log('Reading validation files list')
        for validFile in f.readlines():
            validSet.append(validFile.replace('\n', ''))
else:
    trainSet, validSet, testSet = dataObj.partDataset()
    with open(os.path.join(trainPath, 'trainFileList.txt'), 'w') as f:
        log('Creating training files list')
        for el in trainSet:
            f.write(el+'\n')
    with open(os.path.join(testPath, 'testFileList.txt'), 'w') as f:
        log('Creating test files list')
        for el in testSet:
            f.write(el+'\n')
    with open(os.path.join(validPath, 'validFileList.txt'), 'w') as f:
        log('Creating validation files list')
        for el in validSet:
            f.write(el+'\n')

####################################### RUN DEEP SALIENCE MODEL #############################################
# testSet = testSet[:1]
# print(testSet)
if RUNDEEPSALIENCE:
    # Load and initialize deep salience model
    deepModel = load_model('melody2')
    print(deepModel.summary())
    _ = deepModel.predict(np.zeros((2,360,500,6))) # Initialize model
    # _,_ = predictDeepSalience(path, inputPath, trainSet)
    # _,_ = predictDeepSalience(path, inputPath, validSet)
    # predictions = predictDeepSalience(path, outPath, testSet)
    labels, predictions = getLabels([], dataObj, testSet, params, modelDim, voicing, fftSize, rnnBatch)
    # if not os.path.exists(os.path.join(outPath, 'mel2_targets.npy')):
    # np.save(os.path.join(outPath, 'mel2_targets.npy'), labels.astype(np.float32))
    # np.save(os.path.join(outPath, 'mel2_outputs.npy'), predictions.astype(np.float32))
else:
    # Grab already computed deep salience representation data and compute scores on it
    predictions = np.load(os.path.join(outPath, 'mel2_outputs.npy'))
    labels = np.load(os.path.join(outPath, 'mel2_targets.npy'))
## COMPUTE SCORES FOR DEEP SALIENCE
log('Computing scores for Deep Salience Model')
mysong = []
inp = []
offset = 0
bucketList = dataObj.bucketDataset(testSet, rnnBatch)
for bucket in bucketList:
    longest, _ = dataObj.findLongest(bucket[0])
    for (s, song) in enumerate(sorted(bucket[0])):
        i = glob.glob(os.path.join(dataObj.inputPath, '{}_mel2_input.npy'.format(testSet[l])))
        j = glob.glob(os.path.join(dataObj.targetPath, '{}_mel2_target.npy'.format(testSet[l])))
        if i and j:
            mysong.append([])
            inp.append([])
            for j in range(int(longest/batchSize)):
                mysong[-1].append(labels[offset+j*rnnBatch+s,:,:].T)
                inp[-1].append(predictions[offset+j*rnnBatch+s,:,:].T)
            mysong[-1] = np.hstack(mysong[-1])
            inp[-1] = np.hstack(inp[-1])
            mysong[-1] = mysong[-1].T
            inp[-1] = inp[-1].T
            mask = np.where(mysong[-1]==-1)
            if any(mask[0]):
                mysong[-1] = mysong[-1][0:mask[0][0],:]
                inp[-1] = inp[-1][0:mask[0][0],:]
    offset += int(np.floor(longest/batchSize)) - 1

# mysong = []
# inp = []
# for j in range(int(labels.shape[0]/rnnBatch)):
#     mysong.append(labels[j*rnnBatch,:,:].T)
#     inp.append(predictions[j*rnnBatch,:,:].T)
# mysong = np.hstack(mysong)
# inp = np.hstack(inp)
# mask = np.where(mysong==-1)
# if any(mask[1]):
#     mysong = mysong[:,0:mask[1][0]]
#     inp = inp[:,0:mask[1][0]]

# scores = calculate_metrics(None, [inp.T], [mysong.T], [inp.T], ['AClassicEducation_NightOwl'], binsPerOctave, nOctave, testPath, 1000, 16, thresh=0.05, voicing=True)
# print(scores)

all_scores, _, _, melodyEstimation = calculate_metrics(dataObj, inp, mysong, inp, testSet, binsPerOctave, nOctave, testPath, batchSize, rnnBatch, thresh=0.05, voicing=voicing)
writeScores(all_scores, testPath)

########################################### ADD RNN #######################################################
if TRAIN:
    dataObj = DataSet(inputPath, targetPath, modelDim)
    log('Building temporal models with RNN on Deep Salience predictions')
    ### TRAIN RNN ON DEEP SALIENCE REPRESENTATION
    myModel, modelSplit = trainModel(params, dataObj, trainSet, validSet, modelDim, outPath, voicing, fftSize, rnnBatch)

if TEST:
    ### TEST RNN ON DEEP SALIENCE REPRESENTATION
    preds, labs, realTestSet = test(train, myModel, dataObj, testSet, params, modelDim, targetPath, voicing, fftSize, rnnBatch)

    ### CALCULATE SCORES
    all_scores, melodyEstimation = calculate_metrics(preds, labs, testSet, int(binsPerOctave), int(nOctave), thresh=0.15, voicing=voicing)
    savePlot(preds, labs,realTestSet, testPath)
    writeScores(all_scores, outPath)
    plot(outPath)
    del myModel

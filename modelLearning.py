import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import keras
import keras.backend as K
from keras.utils import Sequence
import glob, os, sys, csv, random
from utils import log, binarize
from data_creation import DataSet, getLabelMatrix
from model import model
from evaluation import calculate_metrics, plotScores, plotThreeScores, writeScores
import time

def bkld(y_true, y_pred):
    """Brian's KL Divergence implementation
    """
    y_true = K.clip(y_true, K.epsilon(), 1.0 - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return K.mean(K.mean(
        -1.0*y_true* K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred),
        axis=-1), axis=-1)

def trainModel(params, dataobj, trainSet, validSet, modelDim, outPath, nUnits, fftSize, rnnBatch):
    K.set_learning_phase(1)
    log('')
    log('---> Training Model <---')
    batchSize = int(params['batchSize'])
    timeDepth = int(params['timeDepth'])
    nHarmonics = int(params['nHarmonics'])
    hopSize = int(params['hopSize'])
    nEpochs = int(params['nEpochs'])
    binsPerOctave = int(params['binsPerOctave'])
    nOctave = int(params['nOctave'])
    log("Number of Epochs: ", nEpochs)
    stateFull = True if params['stateFull']=="True" else False
    myModel, modelSplit = model(modelDim, batchSize, fftSize, timeDepth, nHarmonics, nUnits, stateFull, rnnBatch)
    print(myModel.summary())
    log("Is Statefull: ", stateFull)
    nData, trainSteps, _ = dataobj.sizeDataset(trainSet, batchSize, rnnBatch)
    _, validSteps, _ = dataobj.sizeDataset(validSet, batchSize, rnnBatch)
    nParams = []
    nParams.append([np.prod(np.shape(w)) for w in myModel.get_weights()])
    nParams = np.asarray(np.asarray(nParams).sum()).sum()
    log("Size of Dataset in samples: ", nData)
    nData = nData*fftSize*nHarmonics
    log("Data to Parameters ratio (DPR):", float(nData) / nParams)
    dataGenerator = dataobj.formatDataset(myModel, trainSet, timeDepth, modelDim, batchSize, hopSize, fftSize, nHarmonics, binsPerOctave, nOctave, rnnBatch, stateFull)
    validationGenerator = dataobj.formatDataset(myModel, validSet, timeDepth, modelDim, batchSize, hopSize, fftSize, nHarmonics, binsPerOctave, nOctave, rnnBatch, stateFull)
    filepath = os.path.join(outPath, "weights.{epoch:02d}-{loss:.2f}.hdf5")
    # if "MULTILABEL" not in modelDim:
    myModel.fit_generator(
        generator=dataGenerator,
        steps_per_epoch=trainSteps,
        epochs=int(nEpochs),
        validation_data=validationGenerator,
        validation_steps=validSteps,
        callbacks=[
        keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(patience=10),
        # testModelCb,
        keras.callbacks.EarlyStopping(patience=15, mode='min'),
        # plot_losses
        ],
        verbose = 2,
        shuffle = False
        )
    return myModel, modelSplit

def test(train, myModel, dataobj, testSet, params, modelDim, targetPath, fftSize, rnnBatch):
    K.set_learning_phase(0)
    log('')
    log('---> Testing Model <---')
    cmap = 'hot'
    batchSize = int(params['batchSize'])
    timeDepth = int(params['timeDepth'])
    nHarmonics = int(params['nHarmonics'])
    hopSize = int(params['hopSize'])
    nEpochs = int(params['nEpochs'])
    binsPerOctave = int(params['binsPerOctave'])
    nOctave = int(params['nOctave'])
    stateFull = True if params['stateFull']=="True" else False
    log("Is Statefull: ", stateFull)
    if not train:
        myModel.summary()
    if stateFull or "1D" in modelDim: ### USE DATA GENERATOR ###
        predictGenerator = dataobj.formatDataset(myModel, testSet, timeDepth, modelDim, batchSize, hopSize, fftSize, nHarmonics, binsPerOctave, nOctave, rnnBatch=rnnBatch, stateFull=stateFull)
        nData, testStep, nSongs = dataobj.sizeDataset(testSet, batchSize, rnnBatch)
        preds = myModel.predict_generator(predictGenerator, steps=testStep, verbose=1)
        if "MULTILABEL" in modelDim:
            labNote, labOctave, inputs, trackList = getLabelMatrix(myModel, dataobj, testSet, params, modelDim, fftSize, rnnBatch)
        else:
            labs, inputs, trackList = getLabelMatrix(myModel, dataobj, testSet, params, modelDim, fftSize, rnnBatch)
        if "MULTILABEL" in modelDim:
            return preds, labNote, labOctave, inputs, trackList
        else:
            return preds, labs, inputs, trackList
    else:
        ### GET PREDICTIONS SONG BY SONG ###
        preds = []
        labs = []
        inputs = []
        for track in testSet:
            path = glob.glob(os.path.join(dataobj.inputPath, '{}_mel2_input.npy'.format(track)))
            curInput = np.load(path[0])
            if "BASELINE" in modelDim or "MULTILABEL" in modelDim:
                curInput = curInput.transpose(1, 2, 0)[None,:,:,:]
                curInput = dataobj.deepModel.predict(curInput)
            pred = myModel.predict(curInput[0,:,:].transpose(1,0), batch_size=rnnBatch)
            path = glob.glob(os.path.join(dataobj.targetPath, '{}_mel2_target.npy'.format(track)))
            curTarget = np.load(path[0])
            preds.append(pred)
            labs.append(curTarget)
            inputs.append(curInput)
        ### Return PREDICTIONS, INPUTS,TARGETS and List of Tracks
        return preds, labs, inputs, testSet

def testDummyData(train, myModel, dataobj, testSet, params, modelDim, targetPath, voicing, fftSize, rnnBatch):
    batchSize = int(params['batchSize'])
    stateFull = True if params['stateFull']=="True" else False
    if not train:
        myModel.summary()
    predictGenerator = dataobj.toyData(myModel, testSet, rnnBatch, batchSize, modelDim, fftSize, stateFull)
    nData, testStep, nSongs = dataobj.sizeDataset(testSet, batchSize, rnnBatch)
    print(testStep)
    preds = myModel.predict_generator(predictGenerator, steps=testStep, verbose=1)

    return preds

def zeroPad(data, maxLen, dim):
    shape = np.shape(data)
    newShape = shape
    newShape[dim] = maxLen
    zeroVec = np.zeros((newShape))
    return np.concatenate((data, zeroVec), dim)

################################# SOME CALLBACKS #####################################

class testModel(keras.callbacks.LambdaCallback):
    def __init__(self, model, iterator, steps, path, timeDepth, modelDim, fftSize):
        self.iterator = iterator
        self.model = model
        self.steps = steps
        self.path = path
        self.n = 0
        self.timeDepth = timeDepth
        self.modelDim = modelDim
        self.dim = fftSize

    def on_epoch_end(self, epoch, logs):
        log('Loss on validation:', logs['val_loss'])
        L = int(self.timeDepth)
        if "VOICING" in self.modelDim:
            nOut = self.dim + 1
        else:
            nOut = self.dim
        # ycont = np.zeros((1, nOut))
        # predcont = np.zeros((1, nOut))
        for j in range(0, random.randint(1, self.steps)):
            x, y = next(self.iterator)
        pred = self.model.predict(x)
        plotScores(pred, y)
        self.n += 1

class PlotLosses(keras.callbacks.Callback):
    def __init__(self, nEpochs, outPath):
        self.nEpochs = nEpochs
        self.path = outPath

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

    def on_train_end(self, logs={}):
        # clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.savefig(os.path.join(self.path, 'trainGraph'))

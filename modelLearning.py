import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import keras
import keras.backend as K
from keras.utils import Sequence
import glob, os, sys, csv, random
from utils import log, binarize
from data_creation import toyData, DataSet
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

def trainModel(params, dataset, trainSet, validSet, modelDim, outPath, voicing, fftSize):
    log('')
    log('Training Model')
    locals().update(params)
    batchSize = int(params['batchSize'])
    timeDepth = int(params['timeDepth'])
    nHarmonics = int(params['nHarmonics'])
    hopSize = int(params['hopSize'])
    nEpochs = int(params['nEpochs'])
    myModel, modelSplit = model(modelDim, batchSize, fftSize, timeDepth, nHarmonics)
    log(myModel.summary())
    nData, trainStep, _ = dataset.sizeDataset(trainSet, timeDepth, batchSize, hopSize, modelDim)
    _ , validStep, _ = dataset.sizeDataset(validSet, timeDepth, batchSize,hopSize, modelDim)
    nParams = []
    if modelSplit:
        nParams.append([np.prod(np.shape(w)) for w in convModel.get_weights()])
        nParams.append([np.prod(np.shape(w)) for w in recModel.get_weights()])
        nParams = np.asarray(np.asarray(nParams).sum()).sum()
    else:
        nParams.append([np.prod(np.shape(w)) for w in myModel.get_weights()])
        nParams = np.asarray(nParams).sum()
    log("Size of Dataset in samples: ", nData)
    nData = nData #* fftSize * int(nHarmonics) * timeDepth
    log("Data to Parameters ratio (DPR):", int(nData) / nParams)
    log("Number of training steps: ", trainStep)
    dataGenerator = dataset.formatDataset(trainSet, timeDepth, modelDim, batchSize, hopSize, fftSize, nHarmonics, voicing=voicing)
    validationGenerator = dataset.formatDataset(validSet, timeDepth, modelDim, batchSize, hopSize, fftSize, nHarmonics, voicing=voicing)
    iterator = dataset.formatDataset(trainSet, timeDepth, modelDim, batchSize, hopSize, fftSize, nHarmonics, voicing=voicing)
    filepath = os.path.join(outPath, "weights.{epoch:02d}-{loss:.2f}.hdf5")
    testModelCb = testModel(myModel, iterator, trainStep, outPath, timeDepth, modelDim, fftSize)
    plot_losses = PlotLosses(nEpochs, outPath)
    myModel.fit_generator(
        generator=dataGenerator,
        steps_per_epoch=trainStep,
        epochs=int(nEpochs),
        validation_data=validationGenerator,
        validation_steps=validStep,
        callbacks=[
        keras.callbacks.ModelCheckpoint(filepath, save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(patience=5),
        testModelCb,
        keras.callbacks.EarlyStopping(patience=25, mode='min'),
        plot_losses
        ],
        shuffle=False
        )
    return myModel, modelSplit

def trainFromEpoch(myModel, params, dataset, trainSet, validSet, modelDim, outPath, voicing, fftSize):
    log('')
    log('Training Model')
    locals().update(params)
    nEpochs = int(params['epochs'])
    log(myModel.summary())
    nData, trainStep, _ = dataset.sizeDataset(trainSet, int(timeDepth), int(batchSize),int(hopSize), modelDim)
    _ , validStep, _ = dataset.sizeDataset(validSet, int(timeDepth), int(batchSize),int(hopSize), modelDim)
    nParams = []
    if modelSplit:
        nParams.append([np.prod(np.shape(w)) for w in convModel.get_weights()])
        nParams.append([np.prod(np.shape(w)) for w in recModel.get_weights()])
        nParams = np.asarray(np.asarray(nParams).sum()).sum()
    else:
        nParams.append([np.prod(np.shape(w)) for w in myModel.get_weights()])
        nParams = np.asarray(nParams).sum()
    log("Size of Dataset in samples: ", nData)
    nData = nData * fftSize * int(nHarmonics) * int(timeDepth)
    log("Data to Parameters ratio (DPR):", int(nData) / nParams)
    log("Number of training steps: ", trainStep)
    dataGenerator = dataset.formatDataset(trainSet, int(timeDepth), modelDim, int(batchSize), int(hopSize), fftSize, int(nHarmonics), voicing=voicing)
    validationGenerator = dataset.formatDataset(validSet, int(timeDepth), modelDim, int(batchSize), int(hopSize), fftSize, int(nHarmonics), voicing=voicing)
    iterator = dataset.formatDataset(validSet, int(timeDepth), modelDim, int(batchSize), int(hopSize), fftSize, int(nHarmonics), voicing=voicing)
    filepath = os.path.join(outPath, "weights.{epoch:02d}-{loss:.2f}.hdf5")
    testModelCb = testModel(myModel, iterator, int(trainStep), outPath, timeDepth, modelDim, fftSize)
    history = myModel.fit_generator(
        generator=dataGenerator,
        steps_per_epoch=trainStep,
        epochs=nEpochs,
        validation_data=validationGenerator,
        validation_steps=validStep,
        callbacks=[
        keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1),
        testModelCb,
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
        ],
        shuffle=False
        )
    return myModel, modelSplit

class generator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size,:,:]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size,:,:]

        return np.array(batch_x), np.array(batch_y)

def trainStatefull(params, dataobj, trainSet, validSet, modelDim, outPath, voicing, fftSize, rnnBatch):
    K.set_learning_phase(1)
    log('')
    log('---> Training Statefull Model <---')
    batchSize = int(params['batchSize'])
    timeDepth = int(params['timeDepth'])
    nHarmonics = int(params['nHarmonics'])
    hopSize = int(params['hopSize'])
    nEpochs = int(params['nEpochs'])
    log("Number of Epochs: ", nEpochs)
    stateFull = params['stateFull']
    myModel, modelSplit = model(modelDim, batchSize, fftSize, timeDepth, nHarmonics, False, stateFull, rnnBatch)
    print(myModel.summary())
    log("Is Statefull: ", stateFull)
    nData, trainSteps, _ = dataobj.sizeDataset(trainSet, timeDepth, batchSize, hopSize, fftSize, nHarmonics, modelDim, rnnBatch, stateFull)
    nData, validSteps, _ = dataobj.sizeDataset(validSet, timeDepth, batchSize, hopSize, fftSize, nHarmonics, modelDim, rnnBatch, stateFull)
    nParams = []
    nParams.append([np.prod(np.shape(w)) for w in myModel.get_weights()])
    nParams = np.asarray(np.asarray(nParams).sum()).sum()
    log("Size of Dataset in samples: ", nData)
    nData = nData * fftSize
    log("Data to Parameters ratio (DPR):", float(nData) / nParams)
    count = 0
    patience = 25
    epsilon = 0.1
    trainPath = os.path.join(outPath, 'train')
    validPath = os.path.join(outPath, 'valid')
    trainGraphExamplePath = os.path.join(validPath, 'trainGraphExample')
    validGraphExamplePath = os.path.join(validPath, 'validGraphExample')
    if not os.path.isdir(trainGraphExamplePath):
         os.mkdir(trainGraphExamplePath)
    if not os.path.isdir(validGraphExamplePath):
         os.mkdir(validGraphExamplePath)
    dataGenerator = dataobj.formatDatasetStatefull(trainSet, timeDepth, modelDim, batchSize, hopSize, fftSize, nHarmonics, voicing, rnnBatch=rnnBatch)
    validationGenerator = dataobj.formatDatasetStatefull(validSet, timeDepth, modelDim, batchSize, hopSize, fftSize, nHarmonics, voicing, rnnBatch=rnnBatch)
    # x = np.zeros((1000,500,360))
    # y = np.zeros((1000,500,361))
    # gen = generator(x, y, 10)
    myModel.fit_generator(
        # gen
        generator=dataGenerator,
        steps_per_epoch=trainSteps,
        epochs=int(nEpochs),
        validation_data=validationGenerator,
        validation_steps=validSteps,
        callbacks=[
        keras.callbacks.ModelCheckpoint(outPath, save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(patience=5),
        # testModelCb,
        keras.callbacks.EarlyStopping(patience=25, mode='min'),
        # plot_losses
        ],
        class_weight = None,
        shuffle = False
        )
    return myModel, modelSplit

def test(train, myModel, modelSplit, dataset, testSet, params, modelDim, targetPath, plotTargets, voicing, fftSize, rnnBatch):
    log('')
    log('---> Testing Model <---')
    if rnnBatch == 1:
        singleBatch = True
    else:
        singleBatch = False
    if modelSplit:
        convModel, recModel = myModel
    batchSize = int(params['batchSize'])
    timeDepth = int(params['timeDepth'])
    nHarmonics = int(params['nHarmonics'])
    hopSize = int(params['hopSize'])
    nEpochs = int(params['nEpochs'])
    if not train:
        locals().update(params)
        if modelSplit:
            log(convModel.summary())
            log(recModel.summary())
        else:
            log(myModel.summary())
    _, _, lenTest = dataset.sizeDataset(testSet, int(timeDepth), int(batchSize), int(hopSize), modelDim)
    preds = []
    labs = []
    cnnOut = []
    realTestSet = []
    for k in range(lenTest):
        if isinstance(testSet[k], basestring):
            song = [testSet[k]]
        j = glob.glob(os.path.join(targetPath, '{}_mel1_output.npy'.format(song[0])))
        if any(j):
            realTestSet.append(song)
            if os.path.exists(j[0]):
                predictGenerator = dataset.formatDatasetStatefull(song, int(timeDepth), modelDim, int(batchSize), int(hopSize), fftSize, int(nHarmonics), voicing=voicing, rnnBatch=rnnBatch)
                nSamples, size, length = dataset.sizeDataset(song, int(timeDepth), int(batchSize), int(hopSize), modelDim)
                if nSamples != 0:
                    predictions = None
                    labels = None
                    cnnPred = None
                    inputData = None
                    log("predicting on:"+str(song))
                    log("Size of song in blocks/samples:"+str(size)+'/'+str(nSamples))
                    for l in range(size):
                        one, two, _ = predictGenerator.__next()
                        ### PREDICT WITH MODEL
                        if modelSplit:
                            out = convModel.predict_on_batch(one)
                            # out = binarize(out)
                            ### USE CNN's PREDICTIONS AS INPUTS TO RNN TO BUILD TEMPORAL MODEL
                            newPred = recModel.predict_on_batch(out[None,:,:])
                            if inputData is None:
                                inputData = one[:, :, 0, 0]
                            else:
                                inputData = np.concatenate((inputData, one[:, :, 0, 0]))
                        else:
                            newPred = myModel.predict_on_batch(one)
                        ### SAVE PREDICTIONS AND LABELS TO VARIABLES ###
                        if predictions is None:
                            predictions = newPred
                            labels = two
                            if modelSplit:
                                cnnPred = out
                        else:
                            predictions = np.concatenate((predictions, newPred))
                            labels = np.concatenate((labels, two))
                            if modelSplit:
                                cnnPred = np.concatenate((cnnPred, out))
                        # if "SOFTMAX" in modelDim or "1D" in modelDim:
                        #     predictions = binarize(predictions)
                    ### RE-ARRANGE PREDICTIONS AND LABELS INTO CONTINUOUS MATRIX ###
                    log("SHAPE OF PREDICTIONS:", predictions.shape)
                    if len(labels.shape) == 3:
                        toto = None
                        toto2 = None
                        toto3 = None
                        for i in range(len(labels)):
                            limit = i*labels.shape[1]+labels.shape[1]
                            if limit <= nSamples:
                                if toto is None:
                                    toto = labels[i,:,:]
                                    toto2 = predictions[i,:,:]
                                else:
                                    toto = np.concatenate((toto, labels[i,:,:]), 0)
                                    toto2 = np.concatenate((toto2, predictions[i,:,:]), 0)
                                if modelSplit:
                                    if toto3 is None:
                                        toto3 = cnnPred[i,:,:]
                                    else:
                                        toto3 = np.concatenate((toto3, cnnPred[i,:,:]), 0)
                                lim = np.min((toto.shape[-1], toto2.shape[-1])) # Cut to shortest's length
                        labels = toto[: ,:lim]
                        predictions = toto2[: ,:lim]
                        if modelSplit:
                            cnnPred = toto3[: ,:lim]
                    elif len(labels.shape) == 2:
                        toto = None
                        toto2 = None
                        for l in range(predictions.shape[0]):
                            for k in range(predictions.shape[1]):
                                if toto is None:
                                    toto = predictions[l, k, :, None]
                                    toto2 = labels[l, k, :, None]
                                else:
                                    toto = np.concatenate((toto, predictions[l, k, :, None]), 1)
                                    toto2 = np.concatenate((toto2, labels[l, k, :, None]), 1)
                        predictions = toto.T
                        labels = toto2.T
                    preds.append(predictions)
                    labs.append(labels)
                    if modelSplit:
                        cnnOut.append(cnnPred)

    return preds, labs, realTestSet, cnnOut

def testStatefull(train, myModel, modelSplit, dataobj, testSet, outPath, params, modelDim, targetPath, inputPath, plotTargets, voicing, fftSize, rnnBatch):
    log('')
    log('---> Testing Statefull Model <---')
    cmap = 'hot'
    batchSize = int(params['batchSize'])
    timeDepth = int(params['timeDepth'])
    nHarmonics = int(params['nHarmonics'])
    hopSize = int(params['hopSize'])
    nEpochs = int(params['nEpochs'])
    stateFull = params['stateFull']
    predictGenerator = dataobj.formatDatasetStatefull(testSet, int(timeDepth), modelDim, int(batchSize), int(hopSize), fftSize, int(nHarmonics), voicing, rnnBatch=rnnBatch)
    nData, testStep, nSongs = dataobj.sizeDataset(testSet, timeDepth, batchSize, hopSize, fftSize, nHarmonics, modelDim, rnnBatch, stateFull)
    if not train:
        print(myModel.summary())
    myModel.evaluate_generator(predictGenerator, steps=testStep)
    # realTestSet = []
    # mean_te_acc = []
    # mean_te_loss = []
    # testPath = os.path.join(outPath, 'test')
    # nData, testStep, nSongs = dataobj.sizeDataset(testSet, timeDepth, batchSize, hopSize, fftSize, nHarmonics, modelDim, rnnBatch, stateFull)
    # nRounds = int(np.floor(nSongs/rnnBatch))+1
    # preds = [None] * nSongs
    # inputs = [None] * nSongs
    # labs = [None] * nSongs
    # for s in range(nRounds):
    #     log("Testing on subset {}".format(s))
    #     myModel.reset_states()
    #     lim = min(s*rnnBatch+rnnBatch, nSongs)
    #     subTracks = testSet[s*rnnBatch:lim]
    #     _, subSteps, _ = dataobj.sizeDataset(subTracks, timeDepth, batchSize, hopSize, fftSize, nHarmonics, modelDim, rnnBatch, stateFull)
    #     predictGenerator = dataobj.formatDatasetStatefull(myModel, subTracks, int(timeDepth), modelDim, int(batchSize), int(hopSize), fftSize, int(nHarmonics), voicing=voicing, rnnBatch=rnnBatch)
    #     if voicing:
    #         nOut = int(fftSize)+1
    #     else:
    #         nOut = fftSize
    #     for i in range(subSteps):
    #         batch, targets, newSong = predictGenerator.__next__()
    #         if "MULTILABEL" in modelDim:
    #             targets, targetOctave = targets
    #             testLoss, testAccuracy = myModel.train_on_batch(batch, [targets, targetOctave])
    #         else:
    #             testLoss, testAccuracy = myModel.train_on_batch(batch, targets)
    #         out = myModel.predict(batch)
    #         mean_te_loss.append(testLoss)
    #         mean_te_acc.append(testAccuracy)
    #         for j in range(batch.shape[0]):
    #             ind = s*rnnBatch + j
    #             if ind <= (nSongs - 1):
    #                 if "BASELINE" in modelDim:
    #                     curInput = batch[j,:,:]
    #                 else:
    #                     curInput = batch[j,:,:,5,0]
    #                 if inputs[ind] is None:
    #                     inputs[ind] = curInput
    #                     preds[ind] = out[j]
    #                     labs[ind] = targets[j]
    #                 else:
    #                     inputs[ind] = np.concatenate((inputs[ind], curInput), 0)
    #                     preds[ind] = np.concatenate((preds[ind], out[j]), 0)
    #                     labs[ind] = np.concatenate((labs[ind], targets[j]), 0)
    # if i==subSteps-1:
    #     path = os.path.join(testPath, 'testGraphExample{}'.format(s))
    #     if not os.path.isdir(path):
    #          os.mkdir(path)
    #     for k in range(s*rnnBatch, lim):
    #         try:
    #             if inputs[k] is not None and labs[k] is not None:
    #                 fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    #                 realInput = inputs[k]
    #                 realLabel = labs[k]
    #                 realPred = preds[k]
    #                 mask = realInput.nonzero()
    #                 if any(mask[0]):
    #                     inputs[k] = realInput[0:mask[0][-1],:]
    #                     labs[k] = realLabel[0:mask[0][-1],:]
    #                     preds[k] = realPred[0:mask[0][-1],:]
    #                     ax1.imshow(inputs[k].T, aspect='auto', cmap=cmap, vmax=1, vmin=0)
    #                     ax1.set_title('INPUTS')
    #                     ax2.imshow(labs[k].T, aspect='auto', cmap=cmap, vmax=1, vmin=0)
    #                     ax2.set_title('TARGETS')
    #                     ax3.imshow(preds[k].T, aspect='auto', cmap=cmap, vmax=1, vmin=0)
    #                     ax3.set_title('OUTPUT')
    #                     plt.savefig(os.path.join(path, subTracks[k]))
    #                     plt.close()
    #                 else:
    #                     log("No mask for song:"+str(subTracks[k]))
    #             else:
    #                 log("Input is empty.....")
    #         except Exception as e:
    #             print("Error:"+str(e))
    # log('accuracy testing = {}'.format(np.mean(mean_te_acc)))
    # log('loss testing = {}'.format(np.mean(mean_te_loss)))
    return preds, labs, inputs, realTestSet

def testCNN(train, myModel, dataset, testSet, params, modelDim, targetPath, plotTargets, voicing, fftSize):

    log('')
    log('---> Testing Model <---')
    if not train:
        locals().update(params)
        log(myModel.summary())
    _, _, lenTest = dataset.sizeDataset(testSet, int(timeDepth), int(batchSize), int(hopSize), modelDim)
    labs = []
    realTestSet = []
    cnnOut = []
    for k in range(lenTest):
        if isinstance(testSet[k], basestring):
            song = [testSet[k]]
        j = glob.glob(os.path.join(targetPath, '{}_mel1_output.npy'.format(song[0])))
        if any(j):
            realTestSet.append(song)
            if os.path.exists(j[0]):
                predictGenerator = dataset.formatDataset(song, int(timeDepth), modelDim, int(batchSize), int(hopSize), fftSize, int(nHarmonics), voicing=voicing)
                nSamples, size, length = dataset.sizeDataset(song, int(timeDepth), int(batchSize), int(hopSize), modelDim)
                if nSamples != 0:
                    labels = None
                    cnnPred = None
                    inputData = None
                    log("predicting on:"+str(song))
                    log("Size of song in blocks/samples:"+str(size)+'/'+str(nSamples))
                    for l in range(size):
                        one, two = predictGenerator.__next__()
                        ### PREDICT WITH CNN
                        out = myModel.predict_on_batch(one)
                        # out = binarize(out)
                        if inputData is None:
                            inputData = one[:, :, 0, 0]
                        else:
                            inputData = np.concatenate((inputData, one[:, :, 0, 0]))
                        ### SAVE PREDICTIONS AND LABELS TO VARIABLES ###
                        if cnnPred is None:
                            labels = two
                            cnnPred = out
                        else:
                            labels = np.concatenate((labels, two))
                            cnnPred = np.concatenate((cnnPred, out))
                    ### RE-ARRANGE PREDICTIONS AND LABELS INTO CONTINUOUS MATRIX ###
                    if modelDim=="BASELINE":
                        toto = np.zeros((360, 1))
                        toto3 = np.zeros((360, 1))
                        for i in range(len(labels)):
                            toto = np.concatenate((toto, labels[i,:,:]), 1)
                            toto3 = np.concatenate((toto3, cnnPred[i,:,:]), 1)
                            lim = np.min((toto.shape[-1], toto2.shape[-1])) # Cut to shortest's length
                        labels = toto[: ,1:lim]
                        cnnPred = toto3[: ,1:lim]
                    labs.append(labels)
                    cnnOut.append(cnnPred)

    return cnnOut, labs, realTestSet

def testDeepSalience(dataset, testSet, params, modelDim, targetPath, fftSize):
    log('')
    log('---> Testing Pre-Trained Deep Salience Model <---')
    batchSize = int(params['batchSize'])
    timeDepth = int(params['timeDepth'])
    nHarmonics = int(params['nHarmonics'])
    hopSize = int(params['hopSize'])
    nEpochs = int(params['nEpochs'])
    _, _, lenTest = dataset.sizeDataset(testSet, int(timeDepth), int(batchSize), int(hopSize), int(fftSize), nHarmonics, modelDim)
    labs = []
    pred = []
    realTestSet = []
    for k in range(lenTest):
        if isinstance(testSet[k], str):
            song = [testSet[k]]
        j = glob.glob(os.path.join(targetPath, '{}_mel1_output.npy'.format(song[0])))
        if any(j):
            realTestSet.append(song)
            if os.path.exists(j[0]):
                predictGenerator = dataset.formatDataset(song, int(timeDepth), "rachel", int(batchSize), int(hopSize), int(fftSize), int(nHarmonics), voicing=False)
                nSamples, size, length = dataset.sizeDataset(song, int(timeDepth), int(batchSize), int(hopSize), int(fftSize), nHarmonics, modelDim)
                if nSamples != 0:
                    labels = None
                    inputData = None
                    log("predicting on:"+str(song))
                    log("Size of song in blocks/samples:"+str(size)+'/'+str(nSamples))
                    for l in range(size):
                        one, two = predictGenerator.__next__()
                        for k in range(two.shape[0]):
                            limit = l*two.shape[1]*two.shape[0]+two.shape[1]*k
                            if limit <= nSamples:
                                if inputData is None:
                                    inputData = one[k,:,:]
                                    labels = two[k,:,:]
                                else:
                                    labels = np.concatenate((labels, two[k,:,:]))
                                    inputData = np.concatenate((inputData, one[k,:,:]), 1)
                    print(inputData.shape)
                    labs.append(labels)
                    pred.append(inputData.T)
    return pred, labs, realTestSet, None

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
        ycont = np.zeros((1, nOut))
        predcont = np.zeros((1, nOut))
        for j in range(0, random.randint(1, self.steps)):
            x, y = next(self.iterator)
        pred = self.model.predict(x)
        if len(y.shape)==3:
            for f in range(len(x)):
                ycont = np.concatenate((ycont, y[f, :, :]), 0)
                predcont = np.concatenate((predcont, pred[f, :, :]), 0)
        elif len(y.shape)==2:
            ycont = np.concatenate((ycont, y.T), 1)
            predcont = np.concatenate((predcont, pred.T), 1)
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.imshow(ycont.T, interpolation='nearest', aspect='auto', vmax=1, vmin=0)
        ax1.set_title('Target')
        ax2.imshow(predcont.T, interpolation='nearest', aspect='auto', vmax=1, vmin=0)
        ax2.set_title('Prediction')
        savePath = os.path.join(self.path, 'training_ex_epoch_'+str(self.n))
        plt.savefig(savePath)
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

def zeroPad(data, maxLen, dim):
    shape = np.shape(data)
    newShape = shape
    newShape[dim] = maxLen
    zeroVec = np.zeros((newShape))
    return np.concatenate((data, zeroVec), dim)

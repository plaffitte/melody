import datetime
import sys
import numpy as np

def log(msg, var=None):
    msg = '--->>>' + ' ' + str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ' ' + msg
    if var is not None:
        msg = msg + " " + str(var) + "\n"
    else:
        msg += "\n"
    sys.stdout.flush()
    sys.stdout.write(msg)

def binarize(preds):
    ''' Make matrix binary along first axis
    '''
    if isinstance(preds, list):
        binaryPred = []
        for pred in preds:
            shape = pred.shape
            binaryPred.append([None])
            binaryPred[-1] = np.zeros((shape))
            ind = np.arange(0, shape[0], 1)
            binaryPred[-1][ind, np.argmax(pred, 1)] = 1
    elif len(preds.shape)==2:
        binaryPred = np.zeros_like(preds)
        ind1 = np.arange(0, preds.shape[0], 1, dtype=int)
        ind2 = np.array(np.argmax(preds, 1), dtype=int)
        binaryPred[ind1, ind2] = 1
    elif len(preds.shape)==1:
        if any(preds):
            binaryPred[np.argmax(preds)] = 1

    return binaryPred

def generateDummy(rnnBatch, batchSize, shape, nClasses):
    data = np.zeros((rnnBatch, batchSize, nClasses, shape[0], shape[1]))
    target = np.zeros((rnnBatch, batchSize, nClasses+1))
    for r in range(rnnBatch):
        for b in range(batchSize):
            ind = np.random.randint(1, nClasses)
            data[r, b, ind, :, :] = 1
            target[r, b, ind] = 1

    return data, target

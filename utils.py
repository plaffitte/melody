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
    shape = preds.shape
    binaryPred = np.zeros((shape))
    print("Shape:", shape)
    if len(shape)==3:
        for i in range(shape[0]):
            ind = np.arange(0, shape[1], 1)
            binaryPred[i, ind, np.argmax(preds[i], 1)] = 1
            # binaryPred[i, ind, np.argmax(preds[i, :, 1:], 1)+1] = 1
    elif len(shape)==2:
        ind = np.arange(0, shape[0], 1)
        binaryPred[ind, np.argmax(preds, 1)] = 1
    elif len(shape)==1:
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

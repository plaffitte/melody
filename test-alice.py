'''
This script defines a CNN model and trains it to extract the dominant melody from polyphonic music
'''
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Conv2D, Conv3D, Reshape, Lambda, RNN, LSTM, Flatten
from keras.layers import advanced_activations, pooling, Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import glob
import os
import sys
import numpy as np
from data_creation import toyData, DataSet
from utils import log
import medleydb as mdb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from evaluation import calculate_metrics
import mir_eval

### -------------------->>> PARAM DEFINTION <<<-------------------- ###
fftSize = 360                                                       ###
timeDepth = 10                                                      ###
nHarmonics = 6                                                      ###
# Filters' shape (f, t, h)                                          ###
filterShape1 = [1, 1, 6]                                            ###
filterShape2 = [1, 1, 4]                                            ###
filterShape3 = [1, 1, 2]                                            ###
filterShape4 = [1, 1, 1]                                            ###
filterShape2D = [5, 5, 2]                                           ###
featureMaps1D = 16 # Define the number of 1D feature maps           ###
featureMaps2D = 128                                                 ###
batchSize = 100                                                       ###
nEpochs = 2                                                         ###
MODELDIM = "BASELINE"                                               ###
### ----------------------------------------------------------------###

def model():

    if MODELDIM == "1D":
        input_size = [fftSize, timeDepth, nHarmonics, 1] # Shape of input
        inputs = Input(input_size)
        ### 1D MODEL ###
        layer1 = Conv3D(featureMaps1D, filterShape1, padding='valid', activation='sigmoid')(inputs)
        layer2 = Conv3D(featureMaps1D, filterShape2, padding='valid', activation='sigmoid')(inputs)
        layer3 = Conv3D(featureMaps1D, filterShape3, padding='valid', activation='sigmoid')(inputs)
        layer4 = Conv3D(featureMaps1D, filterShape4, padding='valid', activation='sigmoid')(inputs)
        concatLayer = keras.layers.concatenate([layer1, layer2, layer3, layer4], axis=3)
        reduceDim = Conv3D(1, (10, 10, 10), padding='valid', activation='sigmoid')(concatLayer)
        flatLayer = keras.layers.Flatten()(reduceDim)
        outputDense = Dense(fftSize, activation='sigmoid')(flatLayer)
        interm = Reshape((360, 1))(outputDense)
        outLstm = LSTM(50, activation='sigmoid', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)(interm)
        outputFin = Dense(360, activation='relu')(outLstm)

    elif MODELDIM == "2D":
        input_size = [fftSize, timeDepth, nHarmonics, 1] # Shape of input
        inputs = Input(input_size)
        ### 1D MODEL ###
        layer1 = Conv3D(featureMaps2D, [1, 1, 6], padding='same', activation='sigmoid')(inputs)
        layer2 = Conv3D(1, [5, 5, 1], padding='same', activation='sigmoid')(layer1)
        layer3 = Reshape([360, 10, 6])(layer2)
        layer4 = Conv2D(1, [1, 1], padding='valid', activation='sigmoid')(layer3)
        # outputFin = Lambda(lambda x: K.squeeze(x, axis=3))(layer4)
        outputFin = Reshape([360, 10])(layer4)

    elif MODELDIM == "BASELINE":
        input_shape = (None, None, 6)
        inputs = Input(shape=input_shape)
        # y0 = BatchNormalization()(inputs)
        y1 = Conv2D(128, (5, 5), padding='same', activation='relu', name='bendy1')(inputs)
        # y1a = BatchNormalization()(y1)
        y2 = Conv2D(64 (5, 5), padding='same', activation='relu', name='bendy2')(y1)
        # y2a = BatchNormalization()(y2)
        y3 = Conv2D(64, (3, 3), padding='same', activation='relu', name='smoothy1')(y2)
        # y3a = BatchNormalization()(y3)
        y4 = Conv2D(64, (3, 3), padding='same', activation='relu', name='smoothy2')(y3)
        # y4a = BatchNormalization()(y4)
        y5 = Conv2D(8, (70, 3), padding='same', activation='relu', name='distribute')(y4)
        y5a = BatchNormalization()(y5)
        y6 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='squishy')(y5)
        outputFin = Lambda(lambda x: K.squeeze(x, axis=3))(y6)

    myModel = Model(inputs, outputFin)

    myModel.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])

    return myModel

if __name__ == "__main__":
    expName = sys.argv[1]
    if len(sys.argv)==3:
        train=False
        myModel = keras.models.load_model(sys.argv[2])
    else:
        train=True
    outPath = '/u/anasynth/laffitte/Code/CNNforMelodyExtraction/Experiments'
    inputPath = '/net/vientiane/data/scratch/rmb456/multif0_ismir2017/training_data_with_blur/melody1/inputs'
    targetPath = '/net/vientiane/data/scratch/rmb456/multif0_ismir2017/training_data_with_blur/melody1/outputs'

    pathInputs = os.path.join(os.getcwd(), "inputs.npy")
    pathLabels = os.path.join(os.getcwd(), "targets.npy")
    dataset = DataSet(inputPath, targetPath)
    log('Formatting Training Dataset')
    trainSet, testSet = dataset.partDataset()
    dataGenerator = dataset.formatDataset(trainSet, timeDepth, MODELDIM, batchSize)

    if train:
        ### TRAIN MODEL
        log('')
        log('Training Model')
        myModel = model()
        print(myModel.summary())
        log("Size of Dataset is: ", dataset.sizeDataset(trainSet, timeDepth, batchSize)[0])
        steps = int(np.floor(dataset.sizeDataset(trainSet, timeDepth, batchSize)[0]/batchSize))
        log("Number of training steps: ", steps)
        myModel.fit_generator(dataGenerator, steps_per_epoch=steps , epochs=nEpochs)
        myModel.save(os.path.join(outPath, expName+".h5"))

    ### TEST MODEL
    if not train:
        print(myModel.summary())
    log('')
    log('---> Testing Model <---')
    log("Test set is: " + str(testSet))
    evalGenerator = dataset.formatDataset(testSet, timeDepth, MODELDIM, batchSize)
    sizeTest = int(np.floor(dataset.sizeDataset(testSet, timeDepth, batchSize)[0]/batchSize))
    loss = myModel.evaluate_generator(evalGenerator, steps=sizeTest)
    log("Value of Metrics: " + str(myModel.metrics_names) + " <==> "+ str(loss))
    gen = dataset.formatDataset(testSet, timeDepth, MODELDIM, batchSize)
    labels = []
    counter = 0
    for i in range(sizeTest):
        res = gen.next()
        labels.append(res[1])
        counter += 1
    predictGenerator = dataset.predictionGenerator(testSet, timeDepth, MODELDIM, batchSize)
    predictions = myModel.predict_generator(predictGenerator, steps=sizeTest)
    # predictions = myModel.fit_generator(predictGenerator, steps_per_epoch=sizeTest , epochs=nEpochs)
    log('')
    log('Loss and Accuracy on test set: ', loss)
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.imshow(predictions, interpolation='nearest', aspect='auto')
    ax1.set_title('Network\'s Predictions')
    ax2.imshow(np.reshape(labels, (np.shape(labels)[0]*np.shape(labels)[1], np.shape(labels)[2])), interpolation='nearest', aspect='auto')
    ax2.set_title('Labels')
    print np.shape(predictions), np.shape(np.reshape(labels, (np.shape(labels)[0]*np.shape(labels)[1], np.shape(labels)[2])))
    plt.show()
    plt.savefig(os.path.join(outPath, expName+'result_melody1.png'))
    ### Compute mir metrics
    thresh_vals, thresh_scores = evaluation.get_best_thresh(dat, dataset.trackList, model)
    avg_thresh = [np.mean(thresh_scores[t]) for t in thresh_vals]
    best_thresh = thresh_vals[np.argmax(avg_thresh)]
    all_scores = evaluation.calculate_metrics(preds, dataset.trackList, best_thresh)
    # save scores to data frame
    scores_path = os.path.join(outPath, '_all_scores.csv')
    score_summary_path = os.path.join(outPath, "_score_summary.csv")
    df = pandas.DataFrame(all_scores)
    df.to_csv(scores_path)
    df.describe().to_csv(score_summary_path)
    print(df.describe())

    del myModel

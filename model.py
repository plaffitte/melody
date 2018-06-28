import numpy as np
from utils import *
from keras.layers import Input, Dense, Activation, Conv2D, Conv3D, Reshape, Lambda, LSTM, Flatten, TimeDistributed, Masking, GRU, ConvLSTM2D, Bidirectional
from keras.models import Sequential, Model
from keras.layers import advanced_activations, pooling, Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend as K

def model(modelDim, batchSize, fftSize, timeDepth, nHarmonics, rnnOnly=False, stateFull=False, rnnBatch=16):
    log("Implementing model", modelDim)
    if "CATEGORICAL" in modelDim or "BASELINE":
        loss = ['categorical_crossentropy']
        last_activation = 'softmax'
        metrics = ['categorical_accuracy']
        optimizer = 'nadam'
        nOut = int(fftSize)+1
    else:
        loss = ['binary_crossentropy']
        last_activation = 'sigmoid'
        metrics = ['binary_accuracy']
        optimizer = 'nadam'
        nOut = int(fftSize)
    # if "deepsalience" in modelDim:
    #     nOut = int(fftSize)
    input_shape = [int(fftSize), int(timeDepth), int(nHarmonics)]

    if rnnOnly:
        if "1D" in modelDim or "VOICING" in modelDim:
            recModel = Sequential()
            recModel.add(LSTM(int(fftSize)+1, input_dim=(int(fftSize)+1), input_length=batchSize, activation='sigmoid', stateful=False, return_sequences=True, kernel_initializer='glorot_uniform'))
            recModel.add(TimeDistributed(Dense(int(fftSize)+1)))
            recModel.add(Activation('softmax'))
            recModel.compile(loss=loss, metrics=metrics, optimizer='adam')
            return recModel, True
    else:
        ######################################################################################################
        ################------- RNN MODEL USING DEEP SALIENCE REPRESENTATION INPUTS -------###################
        ######################################################################################################
        if "deepsalience" in modelDim or "BASELINE" in modelDim:
            ### USING RACHEL'S PREDICTIONS AS INPUTS
            myModel = Sequential()
            myModel.add(TimeDistributed(Masking(mask_value=0.), batch_input_shape=[int(rnnBatch), int(batchSize), int(fftSize)]))
            myModel.add(TimeDistributed(BatchNormalization()))
            myModel.add(Bidirectional(GRU(64, activation='linear', stateful=stateFull, return_sequences=True)))
            myModel.add(TimeDistributed(Dense(nOut, activation=last_activation)))
            myModel.compile(loss=loss, metrics=metrics, optimizer=optimizer)
            return myModel, False

        ######################################################################################################
        #############################---------------- 1D MODEL ----------------###############################
        ######################################################################################################
        elif "1D" in modelDim:
            ### USING HOME-GROWN MODEL
            myModel = Sequential()
            myModel.add(TimeDistributed(Masking(mask_value=0.), batch_input_shape=[int(rnnBatch), int(batchSize), int(fftSize), int(timeDepth), int(nHarmonics)]))
            myModel.add(TimeDistributed(BatchNormalization()))
            myModel.add(TimeDistributed(Conv2D(64, (5, 10), padding='same', activation='sigmoid', data_format="channels_last")))
            myModel.add(TimeDistributed(BatchNormalization()))
            myModel.add(TimeDistributed(Conv2D(64, (12, 1), padding='same', activation='sigmoid', data_format="channels_last")))
            myModel.add(TimeDistributed(BatchNormalization()))
            myModel.add(TimeDistributed(Conv2D(1, (1, 10), padding='valid', activation='sigmoid', data_format="channels_last")))
            myModel.add(TimeDistributed(BatchNormalization()))
            myModel.add(TimeDistributed(Flatten()))
            myModel.add(Bidirectional(GRU(nOut, activation='sigmoid', stateful=stateFull, return_sequences=True)))
            myModel.add(TimeDistributed(Dense(nOut, activation=last_activation)))
            myModel.compile(loss=loss, metrics=metrics, optimizer=optimizer)
            return myModel, False

        ######################################################################################################
        #############################---------------- 2D MODEL ----------------###############################
        ######################################################################################################
        elif "2D" in modelDim: ### Model outputting 2D spectral representation with Rachel's model and building temporal model on it with RNN
            if "deepsalience" in modelDim:
                model = Sequential()
                model.add(LSTM(nOut, activation='sigmoid', batch_input_shape=[int(batchSize), int(timeDepth), int(fftSize)], stateful=stateFull, return_sequences=True))
                model.add(TimeDistributed(Dense(nOut)))
                model.add(Activation(last_activation))
                model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
                return model, False
            else:
                ### Model outputting 2D spectral representation with CNN and building temporal model on it with RNN
                ### Note detection
                model = Sequential()
                model.add(BatchNormalization(input_shape=[int(timeDepth), int(fftSize), int(nHarmonics)]))
                ### First filter layer to analyze 5 consecutive frames
                model.add(Conv2D(128, (1, 5), padding='same', activation='sigmoid', data_format="channels_last"))
                model.add(BatchNormalization())
                ### Second filter layer to analyze 12 neighboring frequency bins
                model.add(Conv2D(128, (12, 1), padding='same', activation='sigmoid', data_format="channels_last"))
                model.add(BatchNormalization())
                model.add(Conv2D(1, (1, 1), padding='same', activation='sigmoid', data_format="channels_last"))
                model.add(Lambda(lambda x: K.squeeze(x, axis=3)))
                model.add(BatchNormalization())
                model.add(GRU(256, activation='sigmoid', input_shape=[int(timeDepth), int(fftSize)], stateful=stateFull, return_sequences=True))
                model.add(TimeDistributed(Dense(nOut, activation=last_activation)))
                model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
                return model, False

        ####################################################################################################
        ###########################---------------- CRNN MODEL ----------------#############################
        ####################################################################################################
        elif "CRNN" in modelDim:
            myModel = Sequential()
            myModel.add(TimeDistributed(BatchNormalization(), batch_input_shape=[int(rnnBatch), int(batchSize), int(fftSize), int(timeDepth), int(nHarmonics)]))
            myModel.add(ConvLSTM2D(128, (12, 1), padding='same', activation='sigmoid', data_format="channels_last", return_sequences=True))
            myModel.add(ConvLSTM2D(1, (1 , 20), padding='valid', activation='sigmoid', data_format="channels_last", return_sequences=True))
            myModel.add(TimeDistributed(Flatten()))
            myModel.add(TimeDistributed(BatchNormalization()))
            myModel.add(TimeDistributed(Dense(nOut, activation=last_activation)))
            myModel.compile(loss=loss, metrics=metrics, optimizer=optimizer)
            return myModel, False

        elif "MULTILABEL" in modelDim:
            inputs = Input(batch_input_shape=[int(rnnBatch), int(batchSize), int(fftSize), int(timeDepth), int(nHarmonics)])
            y1 = TimeDistributed(BatchNormalization())(inputs)
            y2 = TimeDistributed(Conv2D(64, (5, 10), padding='same', activation='sigmoid', data_format="channels_last"))(y1)
            y3 = TimeDistributed(BatchNormalization()(y2))
            y4 = TimeDistributed(Conv2D(64, (12, 1), padding='same', activation='sigmoid', data_format="channels_last"))(y3)
            y5 = TimeDistributed(BatchNormalization()(y4))
            y6 = TimeDistributed(Conv2D(1, (1, 10), padding='valid', activation='sigmoid', data_format="channels_last"))(y5)
            y7 = TimeDistributed(BatchNormalization())(y6)
            y8 = TimeDistributed(Flatten())(y7)
            y9a = TimeDistributed(GRU(nOut, activation='sigmoid', stateful=stateFull, return_sequences=True))(y8)
            y9b = TimeDistributed(GRU(nOut, activation='sigmoid', stateful=stateFull, return_sequences=True))(y8)
            outputNote = Dense(12, activation=last_activation)(y9a)
            outputOctave = Dense(6, activation=last_activation)(y9b)
            myModel = Model(inputs, outputs=[outputNote, outputOctave], loss_weights=[1., 1.])
            return myModel, False

def bkld(y_true, y_pred):
    """KL Divergence where both y_true an y_pred are probabilities
    """
    y_true = K.clip(y_true, K.epsilon(), 1.0 - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return K.mean(K.mean(
        -1.0*y_true* K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred),
        axis=-1), axis=-1)

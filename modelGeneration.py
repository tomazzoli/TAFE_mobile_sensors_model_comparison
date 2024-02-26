import tensorflow as tf
from pathlib import Path
from keras.layers import Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from keras import backend as K
from keras.optimizers import SGD, Adam
from constants import *
import sys
class ModelManager:

    def __init__(self,sensor,hyperparams,reTrain=False,dirModelli='modelli'):
        self.__isTrained = False
        self.__hyperparams = hyperparams
        self.__filename = self.__composeFileName(sensor,dirModelli)
        self.__model = self.__initModel(reTrain)

    '''def __generateModel(self):
        my_model = Sequential()
        my_model.add(LSTM(units=self.__hyperparams.nunits, return_sequences=True,
                          input_shape=(self.__hyperparams.timesteps, self.__hyperparams.numfeatures)))
        my_model.add(Dropout(self.__hyperparams.dropout))
        my_model.add(LSTM(units=self.__hyperparams.nunits, return_sequences=True))
        my_model.add(Dropout(self.__hyperparams.dropout))
        my_model.add(Dense(units=1))
        CHOSEN_OPTIMIZATION = SGD(learning_rate=0.01)
        # CHOSEN_LOSS = 'mean_squared_error'
        my_model.compile(optimizer=CHOSEN_OPTIMIZATION, loss=CHOSEN_LOSS, metrics=[MAPE_LABEL, MAE_LABEL, MSE_LABEL])

        return my_model
    '''
    def __generateModel(self):
        my_model = Sequential()
        my_model.add(Bidirectional(LSTM(units=256, return_sequences=True, input_shape=(self.__hyperparams.timesteps, self.__hyperparams.numfeatures))))
        #my_model.add(Dropout(self.__hyperparams.dropout))
        my_model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
        #my_model.add(Dropout(self.__hyperparams.dropout))
        my_model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
        #my_model.add(Dropout(self.__hyperparams.dropout))
        my_model.add(Bidirectional(LSTM(units=32, return_sequences=False)))
        #my_model.add(Dropout(self.__hyperparams.dropout))
        my_model.add(Dense(units=1))
        custom_optimizer = Adam(learning_rate=0.0001)
        #my_model.compile(optimizer=custom_optimizer, loss='mean_absolute_percentage_error', metrics=['mape', 'mae', 'mse'])
        #my_model.compile(optimizer=custom_optimizer, loss='mean_squared_error',
        #                 metrics=['mape', 'mae', 'mse'])
        my_model.compile(optimizer=custom_optimizer, loss='mean_absolute_error',
                         metrics=['mape', 'mae', 'mse'])
        return my_model

    def __initModel(self,reTrain):
        my_file = Path(self.__filename)
        if reTrain:
            my_model = self.__generateModel()
            return my_model
        else:
            if my_file.is_file():
                my_model = tf.keras.models.load_model(self.__filename)
                self.__isTrained = True
                return my_model
            else:
                my_model = self.__generateModel()
                return my_model
    def __composeFileName(self,sensor,dirModelli):
        filename = dirModelli + '/'+ LSTM_MODEL_LABEL + UNDERSCORE + SENSOR_LABEL + UNDERSCORE + str(sensor) + UNDERSCORE + self.__hyperparams.getFileNamePart()
        filename = filename + SUFFISSO_MODELLO_KERAS
        return filename

    def trainModel(self,x_train,y_train):
        history = self.__model.fit(x_train, y_train, epochs=self.__hyperparams.epochs, batch_size=self.__hyperparams.batch_size,
                               validation_split=self.__hyperparams.validation_split)
        self.__model.save(self.__filename)
        self.__isTrained = True

    def getModel(self):
        return self.__model

    def getFileNamePartFromHyperParams(self):
        return self.__hyperparams.getFileNamePart()

    def isModelTrained(self):
        return self.__isTrained

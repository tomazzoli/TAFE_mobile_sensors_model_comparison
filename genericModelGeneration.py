import tensorflow as tf
from pathlib import Path
from keras.layers import Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from keras import backend as K
from keras.optimizers import SGD, Adam
from constants import *
import os
class ModelManager:

    def __init__(self,sensor,hyperparams,reTrain=False,dirModelli='modelli'):
        self.__isTrained = False
        self.hyperparams = hyperparams
        self.modelName = LSTM_MODEL_LABEL
        self.__filename = self.__composeFileName(sensor,dirModelli,modelName=self.modelName)
        self.__model = self.__initModel(reTrain)


    def generateModel(self):
        ''' non da usare, si usano i metodi delle sottoclassi...'''
        pass

    def getModelName(self):
        ''' non da usare, si usano i metodi delle sottoclassi...'''
        pass

    def __initModel(self,reTrain):
        my_file = Path(self.__filename)
        if reTrain:
            my_model = self.generateModel()
            return my_model
        else:
            if my_file.is_file():
                my_model = tf.keras.models.load_model(self.__filename)
                self.__isTrained = True
                return my_model
            else:
                my_model = self.generateModel()
                return my_model
    def __composeFileName(self,sensor,dirModelli,modelName=LSTM_MODEL_LABEL):
        filename = dirModelli + os.path.sep + self.getModelName() + UNDERSCORE + SENSOR_LABEL + UNDERSCORE + str(sensor) + UNDERSCORE + self.hyperparams.getFileNamePart()
        filename = filename + SUFFISSO_MODELLO_KERAS
        return filename

    def trainModel(self,x_train,y_train):
        history = self.__model.fit(x_train, y_train, epochs=self.hyperparams.epochs, batch_size=self.hyperparams.batch_size,
                               validation_split=self.hyperparams.validation_split)
        #self.__model.save(self.__filename)
        self.__isTrained = True

    def getModel(self):
        return self.__model

    def getFileNamePartFromHyperParams(self):
        return self.hyperparams.getFileNamePart()

    def isModelTrained(self):
        return self.__isTrained

    def saveModel(self):
        self.__model.save(self.__filename)

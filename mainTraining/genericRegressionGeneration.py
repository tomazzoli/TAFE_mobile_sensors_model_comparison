from pathlib import Path
from sklearn.linear_model import SGDRegressor
from constants import *
import os
class RegressionManager:

    def __init__(self,sensor):
        self.isTrained = False
        self.sensor = sensor
        self.modelName = "generic unimplemented"
        self.model = self.__initModel()

    def generateModel(self):
        ''' non da usare, si usano i metodi delle sottoclassi...'''
        pass

    def getModelName(self):
        ''' non da usare, si usano i metodi delle sottoclassi...'''
        pass

    def __initModel(self):
        my_model = self.generateModel()
        return my_model

    def trainModel(self,x_train,y_train):
        history = self.model.fit(x_train, y_train)
        self.__isTrained = True

    def getModel(self):
        return self.model

    def isModelTrained(self):
        return self.isTrained

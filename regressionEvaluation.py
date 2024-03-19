import os
import numpy as np
from sklearn.impute import SimpleImputer

from hyperParameters import HyperParameters
from fileManager import FileDataManager
from inputGeneration import DatasetManager
from StochasticGradientDescentRegressionGeneration import StochasticGradientDescentRegressor
from LassoRegressionGeneration import LassoRegressor
from KernelRidgeRegressionGeneration import KernelRidgeRegression
from sklearn.model_selection import train_test_split
from constants import *

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_absolute_error as mae,
                             mean_absolute_percentage_error as mape)


class RegressionEvaluator:
    '''
    ha come parametri il numero del sensore e un array di data_hearders
    '''
    def __init__(self, sensor,data_headers,shuffle):
        self.__sensor = sensor
        self.__csvFileName = self.__generateFileName()
        self.__dataManager = self.__initDataManager(data_headers)
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = self.__create_dataset(shuffle)

        self.__regressionModels = self.__initModels()

    def __generateFileName(self):
        filename = DIR_DATI+os.path.sep+CSV_UNPATCHED_BASE_NAME+str(self.__sensor)+CSV_EXTENSION
        return filename

    def __initDataManager(self,data_headers):
        fm = FileDataManager(self.__csvFileName)
        df = fm.getDataFrame()
        dm = DatasetManager(df, data_headers, data_header_index)
        return dm

    def __create_dataset(self,shuffle):
        dataset = self.__dataManager.getDataArray()
        x = dataset[:,1:43]
        y = dataset[:,-1]
        cleaner = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        cleaner.fit(x,y)
        x = cleaner.transform(x)
        if shuffle:
            [x_train, x_test, y_train, y_test] = train_test_split(x, y, test_size=0.2, random_state=23)
        else:
            [x_train, x_test, y_train, y_test] = train_test_split(x, y, test_size=0.2, random_state=23)

        return x_train, x_test, y_train, y_test

    def __initModels(self,dirModelli=DIR_MODELLI):
        mymodels = {}
        PrimoManager = LassoRegressor(self.__sensor)
        PrimoManager.trainModel(self.__x_train, self.__y_train)
        Primo_model = PrimoManager.getModel()
        Primoname = PrimoManager.getModelName()
        mymodels[Primoname]=Primo_model

        SecondoManager = StochasticGradientDescentRegressor(self.__sensor)
        SecondoManager.trainModel(self.__x_train, self.__y_train)
        Secondo_model = SecondoManager.getModel()
        Secondoname = SecondoManager.getModelName()
        mymodels[Secondoname] = Secondo_model

        TerzoManager = KernelRidgeRegression(self.__sensor)
        TerzoManager.trainModel(self.__x_train, self.__y_train)
        Terzo_model = TerzoManager.getModel()
        altroname = TerzoManager.getModelName()
        mymodels[altroname] = Terzo_model
        ''''''

        return mymodels

    def evaluate(self):
        result = {}
        for key in self.__regressionModels.keys():
            model = self.__regressionModels.get(key)
            predicted = model.predict(self.__x_test)
            sklearn_metrics_mape = mape(self.__y_test, predicted)
            result[key] = sklearn_metrics_mape
        return result

    def getSplittedDataset(self):
        result = {XTRAIN_LABEL:self.__x_train,YTRAIN_LABEL:self.__y_train,XTEST_LABEL:self.__x_test,YTEST_LABEL:self.__y_test}
        return result
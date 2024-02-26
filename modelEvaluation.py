import os
from hyperParameters import HyperParameters
from fileManager import FileDataManager
from inputGeneration import DatasetManager
from modelGeneration import ModelManager
from constants import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_absolute_error as mae,
                             mean_absolute_percentage_error as mape)


class ModelEvaluator:
    '''
    ha come parametri il numero del sensore, gli iperparametri in una istanza della classe HyperParameters e un array di data_hearders
    '''
    def __init__(self, hyperparameters, sensor,data_headers,shuffle):
        self.hyperparameters = hyperparameters
        self.__sensor = sensor
        self.__csvFileName = self.__generateFileName()
        self.__dataManager = self.__initDataManager(data_headers)
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = self.__create_dataset(shuffle)

        self.__model = self.__initModel()

    def __generateFileName(self):
        filename = DIR_DATI+os.path.sep+CSV_BASE_NAME+str(self.__sensor)+CSV_EXTENSION
        return filename

    def __initDataManager(self,data_headers):
        fm = FileDataManager(self.__csvFileName)
        df = fm.getDataFrame()
        dm = DatasetManager(df, data_headers, data_header_index)
        return dm

    def __create_dataset(self,shuffle):
        time_lag = self.hyperparameters.timesteps
        #[x_train, x_test, y_train, y_test] = self.__dataManager.create_input_from_top_shuffle(time_lag)
        if shuffle:
            [x_train, x_test, y_train, y_test] = self.__dataManager.create_input_from_top_shuffle(time_lag)
        else:
            [x_train, x_test, y_train, y_test] = self.__dataManager.create_input_from_top(time_lag)

        return x_train, x_test, y_train, y_test

    def __initModel(self,dirModelli=DIR_MODELLI):
        timesteps = self.__x_train.shape[1]
        numfeatures = self.__x_train.shape[2]
        self.hyperparameters.timesteps = timesteps
        self.hyperparameters.numfeatures = numfeatures
        mm = ModelManager(self.__sensor, self.hyperparameters, reTrain=True,dirModelli=dirModelli)
        if mm.isModelTrained() == False:
            mm.trainModel(self.__x_train, self.__y_train)
        my_model = mm.getModel()
        return my_model

    def evaluate(self):
        batch_size = self.hyperparameters.batch_size
        result = self.__model.predict(self.__x_test, batch_size=batch_size)
        scaler = self.__dataManager.getPredictedNormalizer()
        #predicted = scaler.inverse_transform(result.reshape(len(result), len(result[0])))
        #y_test = scaler.inverse_transform(self.__y_test.reshape(len(self.__y_test), len(self.__y_test[0])))
        predicted = scaler.inverse_transform(result)
        # retrieve only the last element of each sequence (return_sequences=False)
        time_lag =  self.hyperparameters.timesteps
        y_test = scaler.inverse_transform(self.__y_test[0:len(self.__y_test), time_lag - 1])

        sklearn_metrics_mape = mape(y_test, predicted)
        print('sklearn.metrics.mape ', sklearn_metrics_mape)
        return sklearn_metrics_mape,predicted

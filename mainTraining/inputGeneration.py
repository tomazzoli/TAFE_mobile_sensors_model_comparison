import random

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from constants import LOG_SHAPE_OF_MSG,LOG_DATA_WORD_MSG,LOG_TRAIN_WORD_MSG,LOG_VALUES_WORD_MSG

class DatasetManager:

    def __createMultiDimArray(self):
        index = self.__data_header.index(self.__data_header_index)
        data = self.__dataframe[self.__data_header].values
        for i in range(len(data)):
            data[i][index] = data[i][index][len(data[i][index]) - 1]
        data = np.asarray(data).astype(np.float32)

        print(LOG_SHAPE_OF_MSG, LOG_DATA_WORD_MSG, ': ', data.shape)
        return data

    def __createNormalizer(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler

    def __init__(self,dataframe,data_header,data_header_index):
        self.__data_header_index = data_header_index
        self.__data_header = data_header
        self.__dataframe = dataframe
        self.__originaldataarray = self.__createMultiDimArray()
        # normalize the dataset
        self.__scaler = self.__createNormalizer()
        self.__normalizeddataset = self.__scaler.fit_transform(self.__originaldataarray)
        # inverse transform prediction: we need a new mixmaxscaler with the compoments of the original one (numpy features)
        # sklearn.preprocessing.MinMaxScaler has attributes like min_ and scale_
        # we have to transfer these attribute of the last column (the one with the target) to a new empty minmaxscaler
        self.__scaler_pred = self.__createNormalizer()
        target_index = (self.__scaler.n_features_in_.__int__() - 1)
        self.__scaler_pred.min_ = self.__scaler.min_[target_index],
        self.__scaler_pred.scale_ = self.__scaler.scale_[target_index]

    def getDataArray(self):
        return self.__originaldataarray

    def getNormalizer(self):
        return self.__scaler

    def getPredictedNormalizer(self):
        return self.__scaler_pred

    def __create_sequences(self,dataset, seq_length):
        x = []
        y = []
        for i in range(len(dataset)):
            x.append(dataset[i][:seq_length - 1])
            y.append(dataset[i][seq_length - 1:])
        return np.array(x), np.array(y)

    def __create_base_input(self, dataset, time_steps):
        # reshape the data: the input of the LSTM model is a 3D array of shape = (samples, time_steps, features ) where
        # samples = the number of time-series (or sequences)
        # time_steps = the number of instants in each time-series
        # features = the number of elements in each item of the sequence
        x, y = self.__create_sequences(dataset, len(self.__data_header))
        # check if the dimension are compatible!
        samples = int(x.shape[0] / time_steps)
        x = x[:samples * time_steps]
        y = y[:samples * time_steps]
        variables = x.shape[1]
        x = x.reshape(samples, time_steps, variables)
        y = y.reshape(samples, time_steps, 1)
        # print('Shape of x after reshape: ', x.shape)

        return x, y


    #def create_input_with_shuffle(self, timesteps):
    #    x, y = self.__create_base_input(timesteps)
    #    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)
    #    print(LOG_SHAPE_OF_MSG, LOG_TRAIN_WORD_MSG, LOG_DATA_WORD_MSG, ': ', x_train.shape)
    #    print(LOG_SHAPE_OF_MSG, LOG_TRAIN_WORD_MSG, LOG_VALUES_WORD_MSG, ': ', y_train.shape)
    #    return x_train, x_test, y_train, y_test

    def __create_dataset(self,dataset,look_back):
        x, y = self.__create_sequences(dataset, len(self.__data_header))
        dataX = []
        for i in range(len(dataset) - look_back - 1):
            a = x[i:(i + look_back)]
            dataX.append(a)
        return np.array(dataX), np.array(y)

    def create_input_from_top(self, look_back):
        # split into train and test sets
        train_size = int(len(self.__normalizeddataset) * 0.8)
        test_size = len(self.__normalizeddataset) - train_size
        train = self.__normalizeddataset[0:train_size, :]
        # [MOD 2024-01-17] SARA così è sbagliato!!!
        #test = self.__normalizeddataset[test_size:len(self.__normalizeddataset)]
        test = self.__normalizeddataset[len(self.__normalizeddataset)-test_size:]
        # reshape into X=t and Y=t+1
        #trainX, trainY = self.__create_dataset(train, look_back)
        #testX, testY = self.__create_dataset(test, look_back)
        # [2024-01-17] SARA
        trainX, trainY = self.__create_base_input( train, look_back )
        testX, testY = self.__create_base_input( test, look_back)
        # reshape input to be [samples, time steps, features]
        #trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], len(self.__data_header)-1))
        #testX = np.reshape(testX, (testX.shape[0], testX.shape[1], len(self.__data_header)-1))

        return trainX,testX,trainY,testY


    # Sara: random generation of training and test sets
    def create_input_from_top_shuffle(self, look_back):
        # generate the sequences
        x, y = self.__create_base_input(self.__normalizeddataset, look_back)

        # shuffle the lists with same order
        zipped = list(zip(x, y))
        random.Random(40938233124).shuffle(zipped)
        x, y = zip(*zipped)
        x = np.array(x)
        y = np.array(y)

        # split into train and test sets
        train_size = int(len(x) * 0.8)

        trainX = x[0:train_size]
        trainY = y[0:train_size]
        testX = x[train_size:]
        testY = y[train_size:]
        return trainX,testX,trainY,testY








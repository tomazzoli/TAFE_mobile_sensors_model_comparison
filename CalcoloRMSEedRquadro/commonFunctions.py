import random
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from constants import *

def createMultiDimArray(dataframe,data_header,header_index):
    index = data_header.index(header_index)
    data = dataframe[data_header].values
    quanti = len(data)
    for i in range(quanti):
        a = data[i][index]
        data[i][index] = data[i][index][len(data[i][index]) - 1]
    data = np.asarray(data).astype(np.float32)

    print(LOG_SHAPE_OF_MSG, LOG_DATA_WORD_MSG, ': ', data.shape)
    return data

def create_base_input(dataset, time_steps):
    xl = []
    yl = []
    seq_length = len(data_header)
    for i in range(len(dataset)):
        xl.append(dataset[i][:seq_length - 1])
        yl.append(dataset[i][seq_length - 1:])
    x,y = np.array(xl), np.array(yl)
    samples = int(x.shape[0] / time_steps)
    x = x[:samples * time_steps]
    y = y[:samples * time_steps]
    variables = x.shape[1]
    x = x.reshape(samples, time_steps, variables)
    y = y.reshape(samples, time_steps, 1)
    # print('Shape of x after reshape: ', x.shape)

    return x, y

def generateModel(timesteps,numfeatures,dropout):
    my_model = Sequential()
    this_droput = dropout
    my_model.add(Bidirectional(LSTM(units=256, return_sequences=True, input_shape=(timesteps, numfeatures))))
    if(this_droput>0):
        my_model.add(Dropout(this_droput))
    my_model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
    if (this_droput > 0):
        my_model.add(Dropout(this_droput))
    my_model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
    if (this_droput > 0):
        my_model.add(Dropout(this_droput))
    my_model.add(Bidirectional(LSTM(units=32, return_sequences=False)))
    if (this_droput > 0):
        my_model.add(Dropout(this_droput))
    my_model.add(Dense(units=1))
    custom_optimizer = Adam(learning_rate=0.0001)
    #my_model.compile(optimizer=custom_optimizer, loss='mean_absolute_percentage_error', metrics=['mape', 'mae', 'mse'])
    #my_model.compile(optimizer=custom_optimizer, loss='mean_squared_error', metrics=['mape', 'mae', 'mse'])
    my_model.compile(optimizer=custom_optimizer, loss='mean_absolute_error', metrics=['mape', 'mae', 'mse'])
    return my_model

def elaboraSingoloSensore(sensorDataframe,epochs = 400,batch_size = 32,dfTrain=pd.DataFrame(),dfTest=pd.DataFrame(),sensore=1):

    print("dfTrain: ", len(dfTrain), "dfTest:", len(dfTest))
    dfMaxAsArray = createMultiDimArray(sensorDataframe, data_header, data_header_index)
    dfTrainAsArray = createMultiDimArray(dfTrain, data_header, data_header_index)
    dfTestAsArray = createMultiDimArray(dfTest, data_header, data_header_index)
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalizedDF = scaler.fit_transform(dfMaxAsArray)
    normalizedtrain = scaler.transform(dfTrainAsArray)
    normalizedtest = scaler.transform(dfTestAsArray)

    print("dfTrainAsArray: ", len(dfTrainAsArray), "dfTestAsArray:", len(dfTestAsArray))

    time_steps = 5
    dropout = 0
    validation_split = 0.1

    trainX, trainY = create_base_input(normalizedtrain, time_steps)
    testX, testY = create_base_input(normalizedtest, time_steps)

    timesteps = trainX.shape[1]
    numfeatures = trainX.shape[2]
    my_model = generateModel(timesteps, numfeatures, dropout)

    history = my_model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size,
                           validation_split=validation_split, verbose=False)
    result = my_model.predict(testX, batch_size=batch_size)

    samples = int(result.shape[0])
    xr = result[:samples * time_steps]

    scaler_pred = MinMaxScaler(feature_range=(0, 1))
    target_index = (scaler.n_features_in_.__int__() - 1)
    scaler_pred.min_ = scaler.min_[target_index],
    scaler_pred.scale_ = scaler.scale_[target_index]
    predicted = scaler_pred.inverse_transform(result)
    yt = testY.reshape(samples * time_steps, 1)
    y_test = scaler_pred.inverse_transform(yt)

    dfOut = pd.DataFrame(dfTest)
    dfOut = dfOut.iloc[:len(y_test), :]
    dfOut.insert(len(dfTest.keys()), "y_test", y_test, False)
    dfOutPocheColonne = dfOut.loc[:, ["target","y_test","evja_temp"]]
    nomeColTarget = 'target_'+str(sensore)
    nomeColYTEST = 'y_test_' + str(sensore)
    nomeColPredicted = 'predicted_' + str(sensore)
    nomeColEvja = 'evja_temp_' + str(sensore)
    dfOutPocheColonne.rename(columns={'target': nomeColTarget, 'y_test': nomeColYTEST, 'evja_temp': nomeColEvja}, inplace=True)
    dfOutConPred = dfOutPocheColonne.iloc[::time_steps, :]
    dfOutConPred.insert(len(dfOutConPred.keys()), nomeColPredicted, predicted, False)
    result = {"dfOutConPred":dfOutConPred,"nomeColTarget":nomeColTarget,"nomeColYTEST":nomeColYTEST,"nomeColPredicted":nomeColPredicted,"nomeColEvja":nomeColEvja}
    return result

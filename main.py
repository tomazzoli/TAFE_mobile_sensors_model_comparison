import tensorflow as tf
from fileManager import FileDataManager
from inputGeneration import DatasetManager
from modelGeneration import ModelManager
from constants import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_absolute_error as mae,
                             mean_absolute_percentage_error as mape)

#FILE_MODELLO_KERAS = './dati/keras_st4_model.h5'
FILE_DATI_CSV = './dati/Dataset_sens_4.csv'
FILE_TEST_CSV = './dati/Test_sens_4.csv'
FILE_MODELLO_KERAS ='./dati/my_model.h5'
# [2024-01-17] SARA DOPPIO
# time_lag = 7
epochs = 350
time_lag = 6
nunits = 256
thisdropout = 0.2

batch_size = 1
validation_split = 0.1

def defineHyperParams(timesteps, numfeatures):
    hyperparameterValues = {
        BATCH_SIZE_LABEL: batch_size,
        VALIDATION_SPLIT_LABEL: validation_split,
        DROPOUT_LABEL: thisdropout,
        NUNITS_LABEL: nunits,
        NUMFEATURES_LABEL: numfeatures,
        TIME_LAG_LABEL: timesteps,
        EPOCHS_LABEL: epochs
    }

    return hyperparameterValues

def myMain():

    fm = FileDataManager(FILE_DATI_CSV)
    df = fm.getDataFrame()
    dm = DatasetManager(df,data_header,data_header_index)

    [x_train, x_test, y_train, y_test] = dm.create_input_from_top(time_lag)

    timesteps = x_train.shape[1]
    numfeatures = x_train.shape[2]
    hp = defineHyperParams(timesteps,numfeatures)
    mm = ModelManager(FILE_MODELLO_KERAS,hp)
    if mm.isModelTrained() == False:
        mm.trainModel(x_train,y_train)
    my_model = mm.getModel()

    #print(x_test[9])
    result = my_model.predict(x_test, batch_size=batch_size)

    a = result[0];

    # [2024-01-17] SARA
    # predicted = dm.getPredictedNormalizer().inverse_transform(result)
    # y_test = dm.getPredictedNormalizer().inverse_transform(y_test)
    predicted = dm.getPredictedNormalizer().inverse_transform(result.reshape(len(result), len(result[0])))
    y_test = dm.getPredictedNormalizer().inverse_transform(y_test.reshape(len(y_test), len(y_test[0])))
    print('predetto ', predicted)
    print('reale ', y_test)
    vediamo = mape(y_test, predicted)
    print('sklearn.metrics.mape ', vediamo)
    return 'Finito main'

if __name__ == '__main__':
    print('iniziato')
    out = myMain()

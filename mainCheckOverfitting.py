from modelEvaluation import ModelEvaluator
from hyperParameters import HyperParameters
from visualizeOverFitting import OverfittingVisualizer
from constants import *
import csv
import os
import time

SENSORE=3
HISTORY_LABEL = 'history'
HYPERPARAM_LABEL = 'hyperparameter'
NB_START_EPOCHS = 50
default_time_lag = 7
default_epochs = 350
nunits = 256
thisdropout = 0.2
numlayers = 2
batch_size = 1
validation_split = 0.1
# DOPPI PER VELOCITA'
#default_epochs = 50
#default_time_lag = 4
#batch_size = 128
#thisdropout = 0.2
thisdropout = 0.0
batch_size = 32


def generateCSVFileName(nomeFile):
    filename = DIR_ESITI + os.path.sep + nomeFile + CSV_EXTENSION
    return filename

def defineHyperParams(datadescr,time_lag=-1,epochs=-1,dropout=-1,shuffle=True):
    hyperparameterValues = {
        BATCH_SIZE_LABEL: batch_size,
        VALIDATION_SPLIT_LABEL: validation_split,
        DROPOUT_LABEL: thisdropout,
        NUNITS_LABEL: nunits,
        TIME_LAG_LABEL: default_time_lag,
        EPOCHS_LABEL: default_epochs,
        NUMLAYERS_LABEL: numlayers,
        DATADESCR_LABEL:datadescr,
        DATASET_SPLIT_RANDOM_LABEL:shuffle
    }
    if time_lag >0:
        hyperparameterValues[TIME_LAG_LABEL] = time_lag
    if epochs > 0:
        hyperparameterValues[EPOCHS_LABEL] = epochs
    if dropout >=0:
        hyperparameterValues[DROPOUT_LABEL] = dropout
    if shuffle:
        hyperparameterValues[BATCH_SIZE_LABEL] = 64
    else:
        hyperparameterValues[BATCH_SIZE_LABEL] = 32
    result = HyperParameters(hyperparameterValues)
    return result

def myMain(time_lag=-1,epochs=-1,dropout=-1,sensore=SENSORE):
    result = {}
    datadescr = 'base'
    st = time.time()
    result = calcolaMape(datadescr,time_lag=time_lag,epochs=epochs,dropout=dropout,sensore=sensore)
    graficoOverfitting = OverfittingVisualizer(result.get(HYPERPARAM_LABEL),
                                               result.get(BILSTM_MODEL_LABEL),
                                               result.get(DIR_DATI), mape=result.get(MAPE_LABEL))
    history = graficoOverfitting.eval(epochs=epochs,time_lag=time_lag, dropout=dropout)
    result[HISTORY_LABEL] = history
    return result

def calcolaMape(datadescr,time_lag=-1,epochs=-1,dropout=-1,sensore=SENSORE):
    hyperparameterValues = defineHyperParams(datadescr, time_lag=time_lag, epochs=epochs, dropout=dropout,
                                             shuffle=False)
    compare = ModelEvaluator(hyperparameterValues, sensore, data_headers.get(datadescr), hyperparameterValues.shuffle)
    mymodel = compare.getBILSTMModel().get(BILSTM_MODEL_LABEL)
    mape = compare.evaluateSingleModel(mymodel)

    modelFileName = DIR_MODELLI + os.path.sep + 'mape_' + format(mape, ".2f") + UNDERSCORE \
                    + hyperparameterValues.getFileNamePart() + SUFFISSO_MODELLO_KERAS
    mymodel.save(modelFileName)
    dataset = compare.getSplittedDataset()

    dataset = compare.getSplittedDataset()
    result = {BILSTM_MODEL_LABEL: mymodel, HYPERPARAM_LABEL: hyperparameterValues, MAPE_LABEL: mape, DIR_DATI:dataset}
    return result


def writeToCsvFile(to_csv):
    outFile = generateCSVFileName('esitiMisurazioneTempoSingleRun')
    keys = to_csv[0].keys()
    with open(outFile, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(to_csv)




if __name__ == '__main__':
    print('iniziato')
    for epochs in range(3, 4, 1):
        for timelag in range(2, 3, 1):
            for dropout_ in range(0, 1, 1):
                dropout = dropout_ / 10
                valori  = myMain(epochs=epochs,time_lag=timelag,dropout=dropout)
                mape = valori.get(MAPE_LABEL)
                print('eseguito con epoch, timesteps, dropout ',epochs,timelag,dropout)
    print('finito')

from modelEvaluation import ModelEvaluator
from hyperParameters import HyperParameters
from visualizeOverFitting import OverfittingVisualizer
from constants import *
import csv
import os
import time

OUTCSVFILE='esitiMisurazioneHyperParams'
HISTORY_LABEL = 'history'
HYPERPARAM_LABEL = 'hyperparameter'
START_EPOCHS = 50
START_SENSOR =1
END_SENSOR =3
START_TIMESTEPS =2
START_DROPOUT =0
# IPERPARAMETRI DI DEFAULT PER SICUREZZA
default_time_lag = 7
default_epochs = 350
default_nunits = 256
default_dropout = 0.2
default_numlayers = 2
batch_size = 1

validation_split = 0.1
# DOPPI PER VELOCITA'
#default_epochs = 50
#default_time_lag = 4
#batch_size = 128
#thisdropout = 0.2
default_dropout = 0.0
batch_size = 32


def generateCSVFileName(nomeFile):
    filename = DIR_ESITI + os.path.sep + nomeFile + CSV_EXTENSION
    return filename

def defineHyperParams(datadescr,time_lag=-1,epochs=-1,dropout=-0.1,shuffle=True):
    hyperparameterDict = {
        BATCH_SIZE_LABEL: batch_size,
        VALIDATION_SPLIT_LABEL: validation_split,
        DROPOUT_LABEL: default_dropout,
        NUNITS_LABEL: default_nunits,
        TIME_LAG_LABEL: default_time_lag,
        EPOCHS_LABEL: default_epochs,
        NUMLAYERS_LABEL: default_numlayers,
        DATADESCR_LABEL:datadescr,
        DATASET_SPLIT_RANDOM_LABEL:shuffle
    }
    if time_lag >0:
        hyperparameterDict[TIME_LAG_LABEL] = time_lag
    if epochs > 0:
        hyperparameterDict[EPOCHS_LABEL] = epochs
    if dropout >=0:
        hyperparameterDict[DROPOUT_LABEL] = dropout
    if shuffle:
        hyperparameterDict[BATCH_SIZE_LABEL] = 64
    else:
        hyperparameterDict[BATCH_SIZE_LABEL] = 32
    result = HyperParameters(hyperparameterDict)
    return result

def elabora(time_lag=-1,epochs=-1,dropout=-0.1,sensore=START_SENSOR):
    result = {}
    datadescr = 'base'
    st = time.time()
    resultFalse = calcolaMape(datadescr,time_lag=time_lag,epochs=epochs,dropout=dropout,sensore=sensore,shuffle=False)
    resultTrue  = calcolaMape(datadescr,time_lag=time_lag,epochs=epochs,dropout=dropout,sensore=sensore,shuffle=True)
    result[MAPE_LABEL+str(False)] = resultFalse.get(MAPE_LABEL)
    result[MAPE_LABEL + str(True)] = resultTrue.get(MAPE_LABEL)
    return result

def verificaGrafica(resultPrecedente,time_lag):
    graficoOverfitting = OverfittingVisualizer(resultPrecedente.get(HYPERPARAM_LABEL),
                                               resultPrecedente.get(BILSTM_MODEL_LABEL),
                                               resultPrecedente.get(DIR_DATI), mape=resultPrecedente.get(MAPE_LABEL))
    history = graficoOverfitting.eval(epochs=epochs,time_lag=time_lag, dropout=dropout)
    result ={HISTORY_LABEL:history}

def calcolaMape(datadescr,time_lag=-1,epochs=-1,dropout=-0.1,sensore=START_SENSOR,shuffle=False):
    hyperparameterValues = defineHyperParams(datadescr, time_lag=time_lag, epochs=epochs, dropout=dropout,
                                             shuffle=shuffle)
    compare = ModelEvaluator(hyperparameterValues, sensore, data_headers.get(datadescr), hyperparameterValues.shuffle)
    mymodel = compare.getBILSTMModel().get(BILSTM_MODEL_LABEL)
    mape = compare.evaluateSingleModel(mymodel)

    modelFileName = DIR_MODELLI + os.path.sep + 'mape_' + format(mape, ".2f") + UNDERSCORE \
                    + hyperparameterValues.getFileNamePart() + SUFFISSO_MODELLO_KERAS
    mymodel.save(modelFileName)
    dataset = compare.getSplittedDataset()

    result = {BILSTM_MODEL_LABEL: mymodel, HYPERPARAM_LABEL: hyperparameterValues, MAPE_LABEL: mape, DIR_DATI:dataset}
    return result


def writeToCsvFile(to_csv):
    outFile = generateCSVFileName(OUTCSVFILE)
    keys = to_csv[0].keys()
    with open(outFile, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(to_csv)

def myLoop(esiti=[],lastsensor=START_SENSOR,lastepochs=START_EPOCHS,lastTimelag=START_TIMESTEPS,lastdropout=START_DROPOUT):
    to_csv = esiti
    primoTurno = True
    for sensore in range(START_SENSOR, END_SENSOR, 1):
        if (sensore < lastsensor) and primoTurno:
            pass
        else:
            for epochs in range(START_EPOCHS, 600, 50):
                if (epochs < lastepochs) and primoTurno:
                    pass
                else:
                    for timelag in range(START_TIMESTEPS, 11, 1):
                        if (timelag < lastTimelag) and primoTurno:
                            pass
                        else:
                            for dropout_ in range(START_DROPOUT, 6, 1):
                                if (dropout_ < lastdropout) and primoTurno:
                                    pass
                                else:
                                    dropout = dropout_ / 10
                                    out = elabora(epochs=epochs, time_lag=timelag, dropout=dropout, sensore=sensore)
                                    out[SENSOR_LABEL] = sensore
                                    out[EPOCHS_LABEL] = epochs
                                    out[TIME_LAG_LABEL] = timelag
                                    out[DROPOUT_LABEL] = dropout
                                    to_csv.append(out)
                                    writeToCsvFile(to_csv)
                                    print(str(out))
                                    primoTurno = False
                                    print('eseguito con sensore =', sensore, 'epoch= ', epochs, ' timesteps =', timelag, 'dropout=',
                          dropout)
    writeToCsvFile(to_csv)

def readActualStatusCsvFile():
    inFile = generateCSVFileName(OUTCSVFILE)
    esiti_csv = []
    if os.path.isfile(inFile):
        with open(inFile, 'r', newline='') as f:
            dict_reader = csv.DictReader(f)
            for row in dict_reader:
                esiti_csv.append(row)
    return esiti_csv

def lastResult(esitiCsv):
    sensore = START_SENSOR
    epochs = START_EPOCHS
    timelag = START_TIMESTEPS
    dropout = START_DROPOUT
    for row in esitiCsv:
        sensore = int(row.get(SENSOR_LABEL))
        epochs = int(row.get(EPOCHS_LABEL))
        timelag = int(row.get(TIME_LAG_LABEL))
        dropout = round(float(row.get(DROPOUT_LABEL)) * 10)
    return sensore, epochs, timelag, dropout

if __name__ == '__main__':
    print('iniziato')
    esiti_csv = readActualStatusCsvFile()
    sensore,epochs,timelag,dropout = lastResult(esiti_csv)
    print (sensore,epochs,timelag,dropout)
    myLoop(esiti=esiti_csv,lastsensor=sensore,lastepochs=epochs,lastTimelag=timelag,lastdropout=dropout)
    print('finito')

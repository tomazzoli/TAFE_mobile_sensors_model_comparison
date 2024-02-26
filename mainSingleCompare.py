from modelEvaluation import ModelEvaluator
from hyperParameters import HyperParameters
from constants import *
import csv
import os
import time

SENSORE = 4
default_time_lag = 7
default_epochs = 350
nunits = 256
thisdropout = 0.2
numlayers = 2
batch_size = 1
validation_split = 0.1
# DOPPI PER VELOCITA'
default_epochs = 5
default_time_lag = 4
batch_size = 128
thisdropout = 0.2
thisdropout = 0.0
batch_size = 32


def generateCSVFileName(nomeFile):
    filename = DIR_ESITI + os.path.sep + nomeFile + CSV_EXTENSION
    return filename

def defineHyperParams(datadescr,time_lag=-1,epochs=-1,shuffle=True):
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
    if shuffle:
        hyperparameterValues[BATCH_SIZE_LABEL] = 64
    else:
        hyperparameterValues[BATCH_SIZE_LABEL] = 32
    result = HyperParameters(hyperparameterValues)
    return result

def myMain(time_lag=-1,epochs=-1,sensore=SENSORE):
    result = {}
    datadescr = 'base'
    st = time.time()
    hyperparameterValues = defineHyperParams(datadescr, time_lag=time_lag, epochs=-epochs, shuffle=True)
    compare = ModelEvaluator(hyperparameterValues, sensore, data_headers.get(datadescr), hyperparameterValues.shuffle)
    elapsed_time = (time.time() - st)
    tempoShuffle = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    result['tempoShuffle'] = tempoShuffle
    hyperparameterValues.shuffle = defineHyperParams(datadescr, time_lag=time_lag, epochs=-epochs, shuffle=False)
    st = time.time()
    compare = ModelEvaluator(hyperparameterValues, sensore, data_headers.get(datadescr), hyperparameterValues.shuffle)
    elapsed_time = (time.time() - st)
    tempoFromTop = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    result['tempoFromTop'] = tempoFromTop
    return result

def writeToCsvFile(to_csv):
    outFile = generateCSVFileName('esitiMisurazioneTempo')
    keys = to_csv[0].keys()
    with open(outFile, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(to_csv)

if __name__ == '__main__':
    print('iniziato')
    to_csv=[]
    out = {}
    for sensore in range(1,2):#era 1,8
        st = time.time()
        out = myMain(epochs = 3,time_lag = 5,sensore=sensore)
        et = time.time()
        tempo = (et - st) / 60
        out['sensore']= sensore
        to_csv.append(out)
        writeToCsvFile(to_csv)
        print(str(out))

    writeToCsvFile(to_csv)
    print('finito')

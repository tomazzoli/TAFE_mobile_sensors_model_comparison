from modelEvaluation import ModelEvaluator
from hyperParameters import HyperParameters
from constants import *
import csv
import os

SENSORE = 4
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

def execEval(datadescr='-',time_lag=-1,epochs=-1,sensore=SENSORE):
    hyperparameterValues = defineHyperParams(datadescr,time_lag=time_lag,epochs=-epochs,shuffle=True)
    compare = ModelEvaluator(hyperparameterValues, sensore,data_headers.get(datadescr),hyperparameterValues.shuffle)
    sklearn_metrics_mape_shuffle,predicted_shuffle = compare.evaluate()
    print('sklearn.metrics.mape (shuffle) ', sklearn_metrics_mape_shuffle)

    hyperparameterValues.shuffle=False
    compare = ModelEvaluator(hyperparameterValues, sensore, data_headers.get(datadescr),hyperparameterValues.shuffle)
    sklearn_metrics_mape_fromTop, predicted_fromTop = compare.evaluate()

    print('sklearn.metrics.mape (from top) ', sklearn_metrics_mape_fromTop)

    return sklearn_metrics_mape_shuffle,sklearn_metrics_mape_fromTop

def myMain(time_lag=-1,epochs=-1,sensore=SENSORE):
    result = {}
    result['base-shuffle'],result['base-fromTop'] = execEval(datadescr='base',time_lag=time_lag,epochs=-epochs,sensore=sensore)
    #
    # out['esteso'] = execEval(datadescr='esteso')
    #
    #result['meno_tempo-shuffle'],result['meno_tempo-fromTop'] = execEval(datadescr='meno_tempo',time_lag=time_lag,epochs=-epochs,sensore=sensore)
    #result['senza_distanza-shuffle'],result['senza_distanza-fromTop'] = execEval(datadescr='senza_distanza',time_lag=time_lag,epochs=-epochs,sensore=sensore)
    #result['senza_evja-shuffle'],result['senza_evja-fromTop'] = execEval(datadescr='senza_evja',time_lag=time_lag,epochs=-epochs,sensore=sensore)
    return result

def writeToCsvFile(to_csv):
    outFile = generateCSVFileName('esiti4567_timesteps5varieEprochs')
    keys = to_csv[0].keys()
    with open(outFile, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(to_csv)

if __name__ == '__main__':
    print('iniziato')
    to_csv=[]
    out = {}
    for sensore in range(4,8):#era 1,8
        for timesteps in range(5, 6):#era 2,11
            for epochs in range(50, 600,50):#era 50, 600,50
                out = myMain(epochs = epochs,time_lag = timesteps,sensore=sensore)
                out['sensore']= sensore
                out['epochs'] = epochs
                out['timesteps'] = timesteps
                to_csv.append(out)
                writeToCsvFile(to_csv)
        print(str(out))

    writeToCsvFile(to_csv)
    print('finito')

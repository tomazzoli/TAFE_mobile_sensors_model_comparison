from regressionEvaluation import RegressionEvaluator
from fileManager import FileDataManager
from inputGeneration import DatasetManager
from constants import *
import csv
import os

SENSORE = 4
default_time_lag = 5
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

def execEval(datadescr='-',sensore=SENSORE):

    compare = RegressionEvaluator(sensore,data_headers.get(datadescr),shuffle=True)
    sklearn_metrics_mapes_shuffle = compare.evaluate()
    print('sklearn.metrics.mape (shuffle) ', sklearn_metrics_mapes_shuffle)
    return sklearn_metrics_mapes_shuffle


def myMain(sensore=SENSORE):
    result = execEval(datadescr='base',sensore=sensore)
    return result

def writeToCsvFile(to_csv):
    outFile = generateCSVFileName('esitiregressione')
    keys = to_csv[0].keys()
    with open(outFile, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(to_csv)

if __name__ == '__main__':
    print('iniziato')
    to_csv=[]
    out = {}
    for sensore in range(1,8):#era 1,8
        out = myMain(sensore=sensore)
        out['sensore']= sensore
        to_csv.append(out)
        writeToCsvFile(to_csv)
        print(str(out))
    print('finito')

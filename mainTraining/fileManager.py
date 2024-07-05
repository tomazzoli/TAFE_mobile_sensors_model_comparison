import pandas as pd
import numpy as np
import os
from keras.models import load_model
from constants import *
#from modelGeneration import r2_score

class FileDataManager:
    def __init__(self, filename):
        self.__data_frame = self.__load_csv(filename)

    def __load_csv(self,filename):
        df = pd.read_csv(filename)
        return df

    def getDataFrame(self):
        return self.__data_frame

def write_csv(filename,listOfResult):
    df = pd.DataFrame(listOfResult)
    df.to_csv(filename, index=False)

def write_keras_model(outdir,selectedModelWithAttributes):
    modelloLetto = selectedModelWithAttributes.get(THIS_MODEL_LABEL)
    dropout = selectedModelWithAttributes.get(PERFORMANCES_LABEL).get(DROPOUT_LABEL)
    nunits = selectedModelWithAttributes.get(PERFORMANCES_LABEL).get(NUNITS_LABEL)
    epochs = selectedModelWithAttributes.get(PERFORMANCES_LABEL).get(EPOCHS_LABEL)
    sensor = selectedModelWithAttributes.get(PERFORMANCES_LABEL).get(SENSOR_LABEL)
    numlayers = selectedModelWithAttributes.get(PERFORMANCES_LABEL).get(NUMLAYERS_LABEL)
    filename = (LSTM_MODEL_LABEL + UNDERSCORE+ SENSOR_LABEL +UNDERSCORE + str(sensor)
                + UNDERSCORE + NUMLAYERS_LABEL + UNDERSCORE + str(numlayers)
                + UNDERSCORE + NUNITS_LABEL + UNDERSCORE + str(nunits)
                + UNDERSCORE + EPOCHS_LABEL + UNDERSCORE + str(epochs)
                + UNDERSCORE + DROPOUT_LABEL + UNDERSCORE + str(dropout))
    fullpath = outdir + os.path.sep + filename
    modelloLetto.save(fullpath)  # creates a directory with the model, both structure and weights
    del modelloLetto  # deletes the existing model
    return fullpath
#def load_keras_model(filename):
    # returns a compiled model, remember to add custom objects
 #   modelloLetto = load_model(filename, custom_objects={R2_LABEL:r2_score})
  #  return modelloLetto

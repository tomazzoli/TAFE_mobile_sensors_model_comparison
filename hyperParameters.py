from constants import *
class HyperParameters:

    def __init__(self,hyperparameters):
        self.timesteps = hyperparameters.get(TIME_LAG_LABEL)
        self.numfeatures = hyperparameters.get(NUMFEATURES_LABEL)
        self.dropout = hyperparameters.get(DROPOUT_LABEL)
        self.nunits = hyperparameters.get(NUNITS_LABEL)
        self.batch_size = hyperparameters.get(BATCH_SIZE_LABEL)
        self.validation_split = hyperparameters.get(VALIDATION_SPLIT_LABEL)
        self.epochs = hyperparameters.get(EPOCHS_LABEL)
        self.numlayers =  hyperparameters.get(NUMLAYERS_LABEL)
        self.datadescr =  hyperparameters.get(DATADESCR_LABEL)
        self.shuffle = hyperparameters.get(DATASET_SPLIT_RANDOM_LABEL)
    def getFileNamePart(self):
        filenamePart = TWOUNDERSCORE + NUMLAYERS_LABEL+ UNDERSCORE + str(self.numlayers)
        filenamePart = filenamePart + TWOUNDERSCORE + NUNITS_LABEL + UNDERSCORE + str(self.nunits)
        filenamePart = filenamePart + TWOUNDERSCORE + EPOCHS_LABEL + UNDERSCORE + str(self.epochs)
        filenamePart = filenamePart + TWOUNDERSCORE + DROPOUT_LABEL + UNDERSCORE + str(self.dropout)
        filenamePart = filenamePart + TWOUNDERSCORE + DATADESCR_LABEL + UNDERSCORE + str(self.datadescr)
        filenamePart = filenamePart + TWOUNDERSCORE + NUMFEATURES_LABEL + UNDERSCORE + str(self.numfeatures)
        filenamePart = filenamePart + TWOUNDERSCORE + DATASET_SPLIT_RANDOM_LABEL + UNDERSCORE + str(self.shuffle)
        return filenamePart

    def toDict(self):
        hyperparameterValues = {
            TIME_LAG_LABEL: self.timesteps,
            NUMFEATURES_LABEL: self.numfeatures,
            DROPOUT_LABEL: self.dropout,
            NUNITS_LABEL: self.nunits,
            BATCH_SIZE_LABEL: self.batch_size,
            VALIDATION_SPLIT_LABEL: self.validation_split,
            EPOCHS_LABEL: self.epochs,
            NUMLAYERS_LABEL:self.numlayers,
            DATADESCR_LABEL:self.datadescr,
            DATASET_SPLIT_RANDOM_LABEL:self.shuffle
        }

        return hyperparameterValues
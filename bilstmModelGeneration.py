from genericModelGeneration import ModelManager
from keras.layers import Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from constants import *
from keras.optimizers import SGD, Adam

class BILSTMModelManager(ModelManager):

    modelName = LSTM_MODEL_LABEL + 'uou'

    def generateModel(self):
        my_model = Sequential()
        this_droput = self.hyperparams.dropout
        my_model.add(Bidirectional(LSTM(units=256, return_sequences=True, input_shape=(self.hyperparams.timesteps, self.hyperparams.numfeatures))))
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

    def getModelName(self):
        return BILSTM_MODEL_LABEL


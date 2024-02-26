from constants import DIR_ESITI,DIR_GRAFICI,UNDERSCORE,XTRAIN_LABEL,XTEST_LABEL,YTEST_LABEL,YTRAIN_LABEL
import matplotlib.pyplot as plt
import os

NB_START_EPOCHS = 50
LOSS_METRIC_NAME = 'loss'
class OverfittingVisualizer:

    def __init__(self, hyperparameterValues, model, dataset, mape=0):
        self.__model = model
        self.__hyperparams = hyperparameterValues
        self.__dataset = dataset
        self.__mape = mape
        self.__filename = pltfilename = (DIR_ESITI + os.path.sep + DIR_GRAFICI + os.path.sep +'mape_'
                                         + format(self.__mape, ".2f")
                                         + UNDERSCORE + hyperparameterValues.getFileNamePart()+'.png')

    def eval(self,epochs=NB_START_EPOCHS,time_lag=-1, dropout=-1):

        x_train = self.__dataset.get(XTRAIN_LABEL)
        y_train = self.__dataset.get(YTRAIN_LABEL)
        x_test = self.__dataset.get(XTEST_LABEL)
        y_test = self.__dataset.get(YTEST_LABEL)
        batch = self.__hyperparams.batch_size

        history = self.__model.fit(x_train, y_train, epochs=epochs, batch_size=batch, validation_data=(x_test, y_test),
                              verbose=0)
        self.__eval_metric(history,LOSS_METRIC_NAME,epochs=epochs,time_lag=time_lag, dropout=dropout)

    def __eval_metric(self, history, metric_name, epochs=NB_START_EPOCHS,time_lag=-1, dropout=-1):
        '''
        Function to evaluate a trained model on a chosen metric.
        Training and validation metric are plotted in a
        line chart for each epoch.

        Parameters:
            history : model training history
            metric_name : loss or accuracy
        Output:
            line chart with epochs of x-axis and metric on
            y-axis
        '''
        metric = history.history[metric_name]
        val_metric = history.history['val_' + metric_name]
        e = range(1, epochs + 1)
        plt.plot(e, metric, 'bo', label='Train ' + metric_name)
        plt.plot(e, val_metric, 'r^', label='Validation ' + metric_name)
        plt.xlabel('Epoch number')
        plt.ylabel(metric_name)
        plt.title('Comparing training and validation ' + metric_name + ' for timesteps=' + str(
            time_lag) + ' and dropout=' + str(dropout))
        plt.legend()
        plt.savefig(self.__filename)

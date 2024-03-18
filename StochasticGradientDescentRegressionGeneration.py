from genericRegressionGeneration import RegressionManager
from sklearn.linear_model import SGDRegressor
class StochasticGradientDescentRegressor(RegressionManager):

    def generateModel(self):
        model = SGDRegressor()
        return model

    def getModelName(self):
        return "SGDRegressor"


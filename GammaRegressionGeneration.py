from genericRegressionGeneration import RegressionManager
from sklearn.linear_model import Lasso
from sklearn.linear_model import GammaRegressor
class GammaRegression(RegressionManager):

    def generateModel(self):
        model = GammaRegressor()
        return model

    def getModelName(self):
        return "GammaRegressor"


from genericRegressionGeneration import RegressionManager
from sklearn.linear_model import Lasso
class LassoRegressor(RegressionManager):

    def generateModel(self):
        model = Lasso()
        return model

    def getModelName(self):
        return "Lasso"


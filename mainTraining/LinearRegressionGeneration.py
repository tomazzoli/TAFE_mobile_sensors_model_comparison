from genericRegressionGeneration import RegressionManager
from sklearn.linear_model import LinearRegression
class LinearRegressor(RegressionManager):

    def generateModel(self):
        model = LinearRegression()
        return model

    def getModelName(self):
        return "LinearRegression"


from mainTraining.genericRegressionGeneration import RegressionManager
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import HistGradientBoostingRegressor
class StochasticGradientDescentRegressor(RegressionManager):

    def generateModel(self):
        estimators = [
            ('lr', RidgeCV()),
            ('svr', LinearSVR(dual="auto", random_state=42))
            ]
        #model = SGDRegressor()
        model = HistGradientBoostingRegressor()
        return model

    def getModelName(self):
        return "SGDRegressor"


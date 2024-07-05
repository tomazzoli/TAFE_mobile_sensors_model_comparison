from genericRegressionGeneration import RegressionManager
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import GammaRegressor
class KernelRidgeRegression(RegressionManager):

    def generateModel(self):
        model = KernelRidge()
        return model

    def getModelName(self):
        return "KernelRidge"


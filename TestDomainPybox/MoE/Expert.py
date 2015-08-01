import numpy as np

class Expert:
    def __init__(self, dimension_in, dimension_out, location, expertIndex):
        self.weights = np.random.random( (dimension_out, dimension_in) )
        self.location = location
        for i in range(dimension_out):
            for j in range(dimension_in):
                self.weights[i][j] = np.random.uniform(-1,1)

        self.error = 0.0
        self.sum_hs = 0.0
        self.best_weights = self.weights.copy()
        self.best_location = self.location.copy()
        self.index = expertIndex

    def mean(self):
        return np.transpose(self.location)

    def setMean(self, mean):
        self.location = np.transpose(mean)

    def resetError(self):
        self.error = 0.0
        self.sum_hs = 0.0

    def addError(self, hs, error_sqr):
        self.error += hs * error_sqr
        self.sum_hs += hs

    def normalizeError(self):
        if self.sum_hs > 0:
            self.error = self.error / self.sum_hs
        else:
            print "Expert sum_hs is 0. Cannot normalize error"

    def computeExpertyhat(self, x):
        return self.weights.dot( np.transpose(x) )

    def saveBestWeights(self):
        self.best_weights = self.weights.copy()
        self.best_location = self.location.copy()

    def setToBestWeights(self):
        self.weights = self.best_weights.copy()
        self.location = self.best_location.copy()


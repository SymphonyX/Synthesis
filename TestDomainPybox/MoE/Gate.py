
class Gate:
    def outputs(self, experts, x):
        raise NotImplementedError("Outputs method must be implemented by all gates")

    def train(self, training_x, training_y, experts, learningRate):
        raise NotImplementedError("train method must be implemented by all gates")

    def saveBestParams(self):
        raise NotImplementedError("Best params method must be implemented by all gates. Keep tracks of best params")

    def setToBestParams(self):
        raise NotImplementedError("Set best params method must be implemented by all gates. Set best params for gate")







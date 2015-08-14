from Gate import *
import numpy as np
import random
import math

class SoftmaxGate(Gate):
    def __init__(self, num_experts, dimensions_in, mode, training_x):
        self.mode = mode

        #For visualization only
        self.experts_weights = list()
        self.tracking_params = list()

        self.saveBestParams()

    def saveBestParams(self):
        print "-"
        #self.best_params = self.ms.copy()

    def setToBestParams(self):
        print "-"
        #self.ms = self.best_params.copy()

    def outputs(self, experts, x):
        outputs = list()
        for expert in experts:
            uj = expert.mean().dot( np.transpose(x) )
            outputs.append(uj)

        return self._softmaxOutputs(outputs)


    def _softmaxOutputs(self, outputs):
        e = np.exp(np.array(outputs))
        outs = e / np.sum(e)
        return outs.ravel()


    def train(self, training_x, training_y, experts, learningRate):
        randPerm = np.random.permutation(len(training_x))

        for trainIndex in range(len(training_x)):
            index = randPerm[trainIndex]
            x = training_x[index]
            y = training_y[index]

            gateOutputs = self.outputs(experts, x)

            fhs, expertOutputs = [], []
            [expertOutputs.append( expert.computeExpertyhat(x) ) for expert in experts]
            expertsOutputs = np.vstack(expertOutputs).T
            
            final_outputs = expertsOutputs.dot( np.transpose(gateOutputs))

            if self.mode == "comp":
                for j in range(len(experts)):
                    output = expertsOutputs[0][j]

                    fh = math.exp( sum((y - output) ** 2) * -0.5) * gateOutputs[j]
                    fhs.append( fh )

            #update weights and centers
            #TODO check when using multidimensional outputs
            for expertIndex in range(len(experts)):
                expert = experts[expertIndex]
                output = expertsOutputs[0][expertIndex]
                dw = 0; dm = 0
                #update weights && centers
                if self.mode == "coop":
                    #cooperative
                    dw = learningRate * (y - final_outputs) * gateOutputs[expertIndex] * x

                    dm = learningRate * sum((y - final_outputs) * (output - final_outputs)) * gateOutputs[expertIndex] * x
                else:
                    #competitive
                    fh = fhs[expertIndex] / sum(fhs)

                    dw = learningRate * (y - output) * fh * x
                    dm = learningRate * (fh - gateOutputs[expertIndex]) * x

                expert.weights += dw
                expert.setMean( expert.mean() + dm )

        self.experts_weights.append( [e.weights.copy() for e in experts] )
        self.tracking_params.append( [e.mean().copy() for e in experts] )

    def setIterationValues(self, iteration, experts):
        for i, expert in enumerate(experts):
            expert.weights = self.experts_weights[iteration][i]
            expert.setMean( self.tracking_params[iteration][i] )


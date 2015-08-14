import numpy as np
import math
from SoftmaxGate import SoftmaxGate
from GaussianGate import GaussianGate
from Expert import Expert
from Plotter import Plotter

from sklearn.preprocessing import PolynomialFeatures

def FourierFeatures(degree, feat):
    if len(feat.shape) == 1:
        num_feat = feat.shape[0]
        new_feat = np.empty( ((degree+1)**num_feat))

        count = np.zeros( (1, num_feat) )
        feat = feat.reshape( (feat.shape[0], 1) )
        for i in range(new_feat.shape[0]):
            new_feat[i] = math.cos(math.pi * count.dot(feat))

            for j in range(count.shape[1]-1, -1, -1):
                if j == count.shape[1]-1:
                    count[0,j] += 1
                elif count[0,j-1] == degree+1:
                    count[0,j-1] = 0
                    count[0,j] += 1
    else:
        num_feat = feat.shape[1]
        new_feat = np.empty( (feat.shape[0], (degree+1)**num_feat))

        for j in range(new_feat.shape[0]):
            count = np.zeros( (1, num_feat) )
            for i in range(new_feat.shape[1]):
                feat_j = feat[j].reshape( (feat[j].shape[0], 1) )
                new_feat[j][i] = math.cos(math.pi * count.dot(feat_j))

                for k in range(count.shape[1]-1, -1, -1):
                    if k == count.shape[1]-1:
                        count[0,k] += 1
                    elif count[0,k-1] == degree+1:
                        count[0,k-1] = 0
                        count[0,k] += 1
    return new_feat




class MixtureOfExperts:
    learningRate = 0.02
    decay = 0.98

    def _transform_poly_features(self, feat):
        if self.poly_degree > 1:
            polynomial = PolynomialFeatures(self.poly_degree)
            feat = polynomial.fit_transform(feat)
        else:
            if len(feat.shape) > 1:
                ones = np.ones( (feat.shape[0], 1) )
                feat = np.append(ones, feat, 1)
            else:
                ones = np.ones( (1,) )
                feat = np.append(ones, feat, 1)
        return feat

    def _transform_fourier_features(self, feat):
        return FourierFeatures(self.poly_degree, feat)

    def transform_features(self, feat):
        if self.feat_type == "polynomial":
            return  self._transform_poly_features(feat)
        elif self.feat_type == "fourier":
            return self._transform_fourier_features(feat)
        else:
            return feat


    def __init__(self, num_experts, gateType, mode, training_x, training_y, poly_degree=1, feat_type=None):
        self.poly_degree = poly_degree
        self.feat_type = feat_type
        training_x = self.transform_features(training_x)

        dimension_in = training_x.shape[1]
        dimension_out = training_y.shape[1]
        self.gateNet = SoftmaxGate(num_experts, dimension_in, mode, training_x) if gateType == "softmax" else GaussianGate(num_experts, dimension_in, dimension_out, mode, training_x)
        self.training_iterations = 0
        self.experts = list()
        self.numExperts = num_experts
        self.bestError = float("inf")
        for i in range(num_experts):
            location = np.ones( (dimension_in, 1) )
            for j in range(dimension_in):
                if j == 0: continue
                maxx = max(training_x[:,j])
                minx = min(training_x[:,j])
                step = (maxx - minx) / num_experts
                location[j] = (minx + step/2) + step * i
            self.experts.append( Expert(dimension_in, dimension_out, location, i) )

    def computeExpertsOutputs(self, x):
        expertOutputs = list()
        [expertOutputs.append( expert.computeExpertyhat(x) ) for expert in self.experts]
        expertOutputs = np.asarray( expertOutputs ).reshape(1, len(expertOutputs))

        return expertOutputs

    def computeMixtureOutput(self, x):
        expertOutputs = self.computeExpertsOutputs(x)
        gateOutputs = self.gateNet.outputs(self.experts, x)
        finalOutput = expertOutputs.dot( np.transpose(gateOutputs) )

        return finalOutput, expertOutputs

    def testMixture(self, xset, yset, recordErrors=False):

        finalOutputs = list()
        xset = self.transform_features(xset)
        for i, x in enumerate(xset):
            mixPrediction, expPrediction = self.computeMixtureOutput(x)
            finalOutputs.append( mixPrediction )

            ########### Grow netowork ############
            ##### Ramamurti equation 8 ###############
            if isinstance(self.gateNet, GaussianGate) and recordErrors:
                y = yset[i]
                hs = self.gateNet._compute_hs(self.experts, x, y)
                for j, expert in enumerate(self.experts):
                    expert.addError( hs[j], np.linalg.norm(y - expPrediction[0][j])**2 )

        error = 0
        for i in range(len(finalOutputs)):
            error += ((yset[i] - finalOutputs[i])**2) / xset.shape[0]

        return error, finalOutputs 


    def growNetwork(self, sortedExperts, training_x, training_y, test_x, test_y):
        for expert in sortedExperts:
            meanBackup = expert.mean().copy()
            alphaBackup = self.gateNet.alphas[expert.index].copy()
            sigmaBackup = self.gateNet.sigma[expert.index].copy()
            weightsBackup = expert.weights.copy()

            newExpert = Expert(training_x.shape[1], training_y.shape[1], expert.location, self.numExperts)
            newExpert.weights = expert.weights.copy()

            ###########Update experts according to Ramamurti page 5######################
            #############################################################################
            firstMean, secondMean = self.gateNet.find_best_means(expert, training_x, training_y)
            expert.setMean(firstMean)
            newExpert.setMean(secondMean)

            newAlpha =  np.array([alphaBackup / 2.0])
            self.gateNet.alphas = np.vstack( (self.gateNet.alphas, newAlpha) )
            self.gateNet.alphas[expert.index] /= 2.0

            newSigma = np.array( [sigmaBackup] )
            self.gateNet.sigma = np.vstack( (self.gateNet.sigma, newSigma) )


            newMean, oldMean = self.gateNet.weighted_2_means(training_x, training_y, newExpert, expert)
            newExpert.setMean(newMean)
            expert.setMean(oldMean)


            self.experts.append(newExpert)
            ##############################################################################
            #################### Test new network ########################################
            learningRate = self.learningRate

            for i in range(5):
                self.gateNet.train(training_x, training_y, self.experts, learningRate)
                learningRate *= 0.9
                error, prediction = self.testMixture(test_x, test_y)
                if  self.bestError - error > 0.0001:
                    print "Error ", error
                    self._saveParameters(error)
                    self.numExperts += 1
                    print "Adding new expert!"
                    return True

            ###### Revert to previous network########
            expert.setMean(meanBackup)
            expert.weights = weightsBackup
            self.gateNet.alphas[expert.index] = alphaBackup
            self.gateNet.sigma[expert.index] = sigmaBackup
            self.experts.remove(newExpert)
            self.gateNet.sigma = np.delete(self.gateNet.sigma, self.numExperts, 0)
            self.gateNet.alphas = np.delete(self.gateNet.alphas, self.numExperts, 0)
            print "Reversing to previous network"

        return False

    def _saveParameters(self, error):
        #
        # print "Saving best parameters"
        self.bestError = error
        self.gateNet.saveBestParams()
        [e.saveBestWeights() for e in self.experts]


    def trainNetwork(self, training_x, training_y, test_x, test_y, maxIterations):
        training_x = self.transform_features(training_x)
        errors = list()
        keepGrowing = True
        while keepGrowing:
            learningRate = self.learningRate
            iterations = 0
            while True:

                self.gateNet.train(training_x, training_y, self.experts, learningRate)
                learningRate = learningRate * self.decay

                [e.resetError() for e in self.experts]
                last_error, prediction = self.testMixture(test_x, test_y, recordErrors=True)

                print "Error: ", last_error
                if last_error < self.bestError:
                    self._saveParameters(last_error)

                errors.append(last_error)

                error_change = 1 if self.training_iterations < 5 else errors[self.training_iterations-1] - errors[self.training_iterations]
               # print "CHANGE ", error_change
                self.training_iterations += 1
                iterations += 1
                if iterations == maxIterations:
                   print "Max iterations reached"
                   break
                # if error_change < .00001:
                #     print "Min Error"
                #     break

            [e.normalizeError() for e in self.experts]
            errorSorted = sorted(self.experts, key=lambda x:x.error, reverse=True)

            if self.bestError < 0.0001:
                break
            keepGrowing = False#self.growNetwork(errorSorted, training_x, training_y, test_x, test_y)
        print "Final network %d experts" %(len(self.experts))


    def setToBestParams(self):
        self.gateNet.setToBestParams()
        for e in self.experts:
            e.setToBestWeights()



    def visualizePredictions(self, trainingdata, trainingoutput, testdata, testoutput, visMode):
        plotter = Plotter()
        if visMode == "rec":
            plotter.recordTraining(self, trainingdata, trainingoutput, testdata, testoutput)
        else:
            plotter.plotPrediction(self, trainingdata, trainingoutput, testdata, testoutput)
            plotter.plotExpertsPrediction(self, testdata, testoutput)
            plotter.plotExpertsCenters(self, trainingdata, trainingoutput)
            plotter.plotGaussians(self, trainingdata, trainingoutput)










import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
import math
from viztricks import FigureSaver
import numpy as np
from SoftmaxGate import SoftmaxGate
from GaussianGate import GaussianGate
from pylab import *
#from scipy.stats import multivariate_normal

class Plotter:

    def _create3Dmesh(self, x, y, mixtureOfExperts):
        xdata = sorted(x)
        ydata = sorted(y, reverse=True)
        X, Y = np.meshgrid(xdata, ydata)
        Z = np.zeros( X.shape )
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                xy = mixtureOfExperts.transform_features(np.asarray([X[i][j], Y[i][j]]))
                finalOutput, expertsOutputs = mixtureOfExperts.computeMixtureOutput( xy )
                Z[i][j] = finalOutput

        return  X, Y, Z


    def plotPrediction(self, mixtureOfExperts, trainingdata, trainingoutput, testdata, testoutput):
        error, prediction = mixtureOfExperts.testMixture(testdata, testoutput)
        figure = plt.figure()
        three_d = False if trainingdata.shape[1] == 1 else True
        #plot1 = figure.add_subplot(111, projection='3d') if three_d else figure.add_subplot(111)
        plot2 = figure.add_subplot(111, projection='3d') if three_d else figure.add_subplot(111)
        cmap = colormaps()

        if three_d:
            xdat = sorted(trainingdata[:,0])
            ydat = sorted(trainingdata[:,1], reverse=True)
            X, Y = np.meshgrid(xdat, ydat)

            #Quadratic
            #Z = np.log((-2*(X-5)**2 + 50) + (-2*(Y-5)**2 + 50))

            #Peaks
            Z = 3*(1-X)**2*np.exp(-X**2 - (Y+1)**2) - 10*(X/5 - X**3 - Y**5) * np.exp(-X**2 - Y**2) - (1/3)*np.exp(-(X+1)**2 - Y**2)
            #plot1.plot_surface(X, Y, Z, cmap=cmap[3])

            xt, yt, zt = self._create3Dmesh(testdata[:,0], testdata[:,1], mixtureOfExperts)
            plot2.plot_surface(xt, yt, zt, cmap=cmap[7])
        else:
            plot1.scatter(trainingdata[:,0], trainingoutput, c="red")
            plot2.scatter(testdata[:,0], prediction, c='blue')

        string = "Error = %f" %(error)
        ax = plot2.axis()
        x_range = ax[1] - ax[0]
        y_range = ax[3] - ax[2]
        if three_d:
            plot2.text(ax[0] + (x_range / 10), ax[3] - (y_range / 10), 0, string, fontsize = 12)
        else:
            plot2.text(ax[0] + (x_range / 10), ax[3] - (y_range / 10), string, fontsize = 12)

        plt.show()

    def recordTraining(self, mixtureOfExperts, trainingdata, trainingoutput, testdata, testoutput):    
        with FigureSaver(name="capture", mode='gif', fps=5):
            for iteration in range(mixtureOfExperts.training_iterations):
                mixtureOfExperts.gateNet.setIterationValues(iteration, mixtureOfExperts.experts)

                error, prediction = mixtureOfExperts.testMixture(testdata, testoutput)
                figure = plt.figure()
                three_d = False if trainingdata.shape[1] == 2 else True
                plot1 = figure.add_subplot(111, projection='3d') if three_d else figure.add_subplot(111)
                cmap = colormaps()

                if three_d:
                    xdat = sorted(trainingdata[:,0])
                    ydat = sorted(trainingdata[:,1], reverse=True)
                    X, Y = np.meshgrid(xdat, ydat)
                    Z = np.log((-2*(X-5)**2 + 50) + (-2*(Y-5)**2 + 50))
                    plot1.plot_surface(X, Y, Z, cmap=cmap[3])

                    xt, yt, zt = self._create3Dmesh(testdata[:,1], testdata[:,2], mixtureOfExperts)
                    plot1.plot_surface(xt, yt, zt, cmap=cmap[7])
                else:
                    plot1.scatter(trainingdata[:,0], trainingoutput, c="red")
                    plot1.scatter(testdata[:,0], prediction, c='blue')

                string = "Error = %f" %(error)
                ax = plot1.axis()
                x_range = ax[1] - ax[0]
                y_range = ax[3] - ax[2]
                if three_d:
                    plot1.text(ax[0] + (x_range / 10), ax[3] - (y_range / 10), 0, string, fontsize = 12)
                else:
                    plot1.text(ax[0] + (x_range / 10), ax[3] - (y_range / 10), string, fontsize = 12)

                plt.show()
                figure.close()

    def plotExpertsPrediction(self, mixtureOfExperts, testdata, testoutput):
        predictions, y = [], []
        for e in mixtureOfExperts.experts:
            predictions.append( list() )

        for x in testdata:
            x = mixtureOfExperts.transform_features(x)
            y.append( mixtureOfExperts.computeMixtureOutput(x)[0] )

            yhats = mixtureOfExperts.computeExpertsOutputs(x)
            for i in range(len(mixtureOfExperts.experts)):
                predictions[i].append( yhats[0][i] )

        plt.subplot(121)
        for i in range(len(predictions)):
            line, = plt.plot(testdata[:,0], predictions[i])
            line.set_label("Expert %d" %(i))

        xs = testdata[:,0]
        line, = plt.plot(xs, y, 'ro')
        line.set_label("Overall Output")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
        plt.show()



    def plotExpertsCenters(self, mixtureOfExperts, trainingdata, trainingoutput):
        figure = plt.figure()
        three_d = False if trainingdata.shape[1] == 1 else True
        means = np.zeros( (len(mixtureOfExperts.experts), mixtureOfExperts.experts[0].location.shape[0]) )

        if three_d:
            plot1 = figure.add_subplot(111)
            labels = list()
            for i in range(len(mixtureOfExperts.experts)):
                expertMean = mixtureOfExperts.experts[i].location
                means[i] = expertMean
                plot1.scatter(expertMean[1], expertMean[2], c="red")
                labels.append("Expert %d center" %(i))

            for label, x, y in zip(labels, means[:, 1], means[:, 2]):
                plt.annotate(
                label,
                xy = (x, y), xytext = (-20, 20),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

        else:
            plot1 = figure.add_subplot(121)
            ax = plot1.axis()
            plot1.scatter(trainingdata[:,0], trainingoutput, c="red")
            ys = np.linspace(min(trainingoutput), max(trainingoutput), num=50)

            for i in range(len(mixtureOfExperts.experts)):
                expertMean = [ mixtureOfExperts.experts[i].location[1] ] * 50
                line, = plot1.plot(expertMean, ys)
                line.set_label("Expert %d center" %(i))

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
        plt.show()

    def plotGaussians(self, mixtureOfExperts, trainingdata, trainingoutput):
        if (isinstance(mixtureOfExperts.gateNet, GaussianGate)):
            three_d = False if trainingdata.shape[1] == 1 else True
            if three_d:
                figure = plt.figure()
                stepfirst = (max(trainingdata[:,0]) - min(trainingdata[:,0])) / 100.0
                stepsecond = (max(trainingdata[:,1]) - min(trainingdata[:,1])) / 100.0
                x, y = np.mgrid[min(trainingdata[:,0]):max(trainingdata[:,0]):stepfirst, min(trainingdata[:,1]):max(trainingdata[:,1]):stepsecond]
                pos = np.empty(x.shape + (2,))
                pos[:, :, 0] = x; pos[:, :, 1] = y
                ax2 = figure.add_subplot(111)
                for i, expert in enumerate(mixtureOfExperts.experts):
                    rv = multivariate_normal(expert.location[1:], mixtureOfExperts.gateNet.sigma[i,1:,1:])
                    ax2.contourf(x, y, rv.pdf(pos),alpha=0.2)
            else:
                figure = plt.figure()
                plot1 = figure.add_subplot(121)
                min_out = min(trainingoutput)
                max_out = max(trainingoutput)
                trainingoutput = [(x - min_out) / (max_out- min_out) for x in trainingoutput]

                plot1.scatter(trainingdata[:,0], trainingoutput, c="red")
                ax = plot1.axis()
                xs = np.linspace(ax[0], ax[1], 100)
                for i in range(len(mixtureOfExperts.experts)):
                    mean = mixtureOfExperts.experts[i].mean()[1]
                    sigma = math.sqrt( mixtureOfExperts.gateNet.sigma[i][1][1] )
                    line, = plot1.plot(xs, mixtureOfExperts.gateNet.alphas[i] * mlab.normpdf(xs, mean, sigma))
                    line.set_label("Expert %d gaussian" %(i))

                plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
            plt.show()











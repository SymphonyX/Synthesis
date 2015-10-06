import sys

sys.path.insert(0, "../../")
sys.path.insert(1, "../MoE")

import pickle
import os
import math
from MixtureOfExperts import *
from dmp import DMP
from optparse import OptionParser

import matplotlib.pyplot as plt

import numpy as np

def normalize_data(data):
    test = data.copy()
    dimensions = len(data.shape)
    if dimensions == 1:
        test = (data - min(data)) / (max(data) - min(data))
        
        mins = [ min(data) ]
        maxs = [ max(data) ]
    else:
        mins, maxs = [], []
        for col in range(data.shape[1]):
            minx = min(data[:,col])
            maxx = max(data[:,col])
            test[:,col] = (data[:,col] - minx) / (maxx - minx) if maxx != minx else 1.0
            
            mins.append(minx)
            maxs.append(maxx)
    return test, mins, maxs


def generateTrainingData(filename):
    data, out = readFile(filename)

    training_x = np.asarray(data)
    training_y = np.asarray(out)

    training_x, xmin, xmax = normalize_data(training_x)
    training_y, ymin, ymax = normalize_data(training_y)

    if len(training_x.shape) == 1:
        training_x = training_x.reshape( (training_x.shape[0], 1) )
    if len(training_y.shape) == 1:
        training_y = training_y.reshape( (training_y.shape[0], 1) )

    return training_x, training_y, np.asarray(xmin), np.asarray(xmax), np.asarray(ymin), np.asarray(ymax)


def readFile(filename):
    f = open(filename, "r")
    lines = f.readlines()
    data = list(); out = list()

    for line in lines[1:]:
        line = line.rstrip('\n')
        split = line.split(';')
        split_data = [float(x) for x in split[0].split(",")]
        split_output = [float(x) for x in split[1].split(",")]

        if len(split_data) > 1:
            data.append( tuple(split_data) )
        elif len(split_data) == 1:
            data.append(split_data[0])

        if len(split_output) > 1:
            out.append( tuple(split_output) )
        elif len(split_output) == 1:
            out.append(split_output[0])

    f.close()
    return np.asarray(data), np.asarray(out)

if __name__ == "__main__":

    parser = OptionParser("usage: %prog [options] arg")
    parser.add_option("-t", "--train", action="store", help="data set", type="string")
    parser.add_option("-l", "--load", action="store", help="load network", type="string")
    parser.add_option("-s", "--save", action="store", help="save network", type="string")
    parser.add_option("-n", "--num", action="store", help="num experts", type="int")

    (options, args) = parser.parse_args()

    if options.train is not None:   
        datain, dataout = readFile(options.train)
    else:
        datain, dataout = readFile("../TrainingTest/train.txt")



    if options.load is not None:
        f = open(options.load, "r")
        data = pickle.load(f)
        networks = data[0]
        xmin = data[1]
        xmax = data[2]
        ymin = data[3]
        ymax = data[4]
    else:
        experts = 1 if options.num is None else options.num
        learningRate = 0.01
        decay = 0.98

        training_x, training_y, xmin, xmax, ymin, ymax = generateTrainingData(options.train)

        networks = []

        for i in range(training_y.shape[1]):
            print "Network ", i
            indexes = np.random.permutation(training_x.shape[0])
            training_xi = training_x[indexes]
            training_yi = training_y[indexes]

            training_yi = training_yi[:,i]
            training_yi = training_yi.reshape( (training_yi.shape[0], 1) )

            mixExperts = MixtureOfExperts(experts, "em", "coop", training_xi, training_yi, poly_degree=1, feat_type="polynomial")
            mixExperts.learningRate = learningRate
            mixExperts.decay = decay


            test_x = training_xi[:training_xi.shape[0]/4]
            train_x = training_xi[training_xi.shape[0]/4:]
            test_y = training_yi[:training_yi.shape[0]/4]
            train_y = training_yi[training_yi.shape[0]/4:]

            mixExperts.trainNetwork(train_x, train_y, test_x, test_y, 30)

            mixExperts.setToBestParams()
            networks.append(mixExperts )


        f = open(options.save, "w") 
        pickle.dump((networks, xmin, xmax, ymin, ymax), f)



    parameters = []
    actual_parameters = []

    for i, input in enumerate(datain):
        x = input
        feat = np.zeros( (1, 1) )
        feat[0,0] = (x - xmin[0]) / (xmax[0] - xmin[0])

        all_params = []
        for j, network in enumerate(networks):
            new_feat = network.transform_features(feat)
            prediction, expertsPrediction = network.computeMixtureOutput(new_feat)
            prediction = (prediction * (ymax[j] - ymin[j])) + ymin[j]
            print "Ex ", j, " Prediction: ", prediction
            all_params.append( prediction )

        actual_parameters.append( dataout[i] )
        parameters.append( all_params )

    for i in range(len(actual_parameters[0])):
        params = []
        for p in actual_parameters:
            params.append(p[i])

        plt.plot(datain, params, label="training")

        params = []
        for p in parameters:
            params.append(p[i][0])

        plt.plot(datain, params, label="prediction")
        plt.legend(bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0.)
        plt.show()
   






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
import Plotter

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
        print "You need a network dude!"
        sys.exit(-1)


    for n, network in enumerate(networks):

        experts_out = []
        actual_ouputs = []
        for i, input in enumerate(datain):
            x = input
            y = dataout[i][n].reshape( (1, 1) ) if len(networks) > 1 else dataout[i].reshape( (1, dataout[i].shape[0]) )
            actual_ouputs.append(y)
            
            feat = np.zeros( (1, 1) )
            feat[0,0] = (x - xmin[0]) / (xmax[0] - xmin[0])
            new_feat = network.transform_features(feat)

            output, expertOutputs = network.computeMixtureOutput(new_feat)

            experts_out.append( expertOutputs )

        print "Total experts ", len(network.experts)
        for j, expert in enumerate(network.experts):

            parameters = []            
            for k, p in enumerate(experts_out):
                parameters.append( p[0,j] )
            
            plt.plot(datain, parameters, label="expert " + str(j) )

        if len(networks) > 1:
            plt.plot(datain, dataout[:,n], linewidth=3.0, label="Actual output")
            for j in range(experts_out[0].shape[1]):
                pred = []
                for p in experts_out:
                    pred.append( p[0,j] )
                plt.plot(datain, pred, label="Expert " + str(j) + " prediction")
            plt.show()
        else:
            for k in range(dataout.shape[1]):
                plt.plot(datain, dataout[:,k], linewidth=3.0, label="Actual output")
                for j in range(experts_out[0].shape[1]):
                    pred = []
                    for p in experts_out:
                        pred.append( p[k,j] )
                    plt.plot(datain, pred, label="Expert " + str(j) + " prediction")

                plt.show()




import sys

sys.path.insert(0, "../")
sys.path.insert(1, "./MoE")

import pickle
import os
import math
from MixtureOfExperts import *
from dmp import DMP

import matplotlib.pyplot as plt

import numpy as np

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print "Arg2: dataset directory"

    f = open("network_parameters.pkl", "r")
    data = pickle.load(f)
    networks = data[0]
    xmin = data[1]
    xmax = data[2]
    ymin = data[3]
    ymax = data[4]

    files = os.listdir(sys.argv[1])
    pkls, thetas = [], []

    for f in files:
        fi = f.split(".")
        if fi[-1] == "pkl":
            pkls.append(f)
            thetas.append( float(fi[0][0]+"."+fi[0][1]) )

    pkls.sort()
    thetas.sort()

    parameters = []
    actual_parameters = []

    for theta in thetas:
        y = 200 * math.sin(theta)
        x = 200 * math.cos(theta)
        feat = np.zeros( (1, 2) )
        feat[0,0] = (x - xmin[0]) / (xmax[0] - xmin[0])
        feat[0,1] = (y - xmin[1]) / (xmax[1] - xmin[1])

        all_params=[]
        for i, network in enumerate(networks):
            new_feat = network.transform_features(feat)
            prediction, expertsPrediction = network.computeMixtureOutput(new_feat)
            prediction = (prediction * (ymax[i] - ymin[i])) + ymin[i]
            all_params.append(prediction)

        parameters.append(all_params)

        t_str = str(theta)
        t_str = t_str.split(".")
        t_str = t_str[0] + t_str[1] + ".pkl"

        params_file = open(sys.argv[1]+t_str, "r")
        params = pickle.load(params_file)

        actual_parameters.append( params )


    for i in range(len(actual_parameters[0])):
        params = []
        for p in actual_parameters:
            params.append(p[i])
        plt.plot(thetas, params, label="training")

        params = []
        for p in parameters:
            params.append(p[i])

        plt.plot(thetas, params, label="prediction")
        plt.legend(bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0.)
        plt._show()






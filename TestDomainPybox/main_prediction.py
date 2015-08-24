import pygame
import sys
sys.path.insert(0, "../")
sys.path.insert(1, "./MoE")

from dmp import DMP
from optparse import OptionParser
from MixtureOfExperts import *
import numpy as np
from domain import ResetWorld
from domain import RunSimulation
import math
import matplotlib.pyplot as plt
import pickle

tau = 2.0
basis = 5

width = 1000
height = 1000

FPS = 60
dt = 1.0 / FPS
origin = (width / 2 + 120, (height / 4)*3 - 350)
dmp_dt = 0.001
fpsClock = None


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

    return training_x, training_y, xmin, xmax, ymin, ymax

def trainNetworks(options):
    experts = 1 if options.num is None else options.num

    learningRate = 0.01 if options.rate is None else options.rate
    decay = 0.98 if options.decay is None else options.decay

    training_x, training_y, xmin, xmax, ymin, ymax = generateTrainingData(options.train)

    networks = []
    for i in range(training_y.shape[1]):
        print "Training network ", i, "\n"

        training_yi = training_y[:,i].reshape( (training_y.shape[0], 1))

        indexes = np.random.permutation(training_x.shape[0])
        training_xi = training_x[indexes]
        training_yi = training_yi[indexes]

        best_model_error = float("inf")
        best_model = None
        # for degree in range(1, 4):

            # sum_errors = 0
            # for i in range(5):
        mixExperts = MixtureOfExperts(experts, "em", "coop", training_x, training_yi, poly_degree=1, feat_type="polynomial")
        mixExperts.learningRate = learningRate
        mixExperts.decay = decay


        test_x = training_xi[:training_xi.shape[0]/4]
        train_x = training_xi[training_xi.shape[0]/4:]
        test_y = training_yi[:training_yi.shape[0]/4]
        train_y = training_yi[training_yi.shape[0]/4:]

        # print "\n\nCross validation k: ", i, "-1\n\n"
        # mixExperts.training_iterations = 0
        # mixExperts.bestError = float("inf")
        mixExperts.trainNetwork(train_x, train_y, test_x, test_y, 30)
            # sum_errors += mixExperts.bestError

                # print "\n\nCross validation k: ", i, "-2\n\n"
                # mixExperts.training_iterations = 0
                # mixExperts.bestError = float("inf")
                # mixExperts.trainNetwork(test_x, test_y, train_x, train_y, 20)
                # sum_errors += mixExperts.bestError

            # if sum_errors < best_model_error:
            #     best_model = mixExperts
            #     best_model_error = sum_errors
            #     print "New best model: degree ", degree
            #     print "Error: ", best_model_error, "\n\n\n"
            #

        mixExperts.setToBestParams()
        networks.append(mixExperts )

    return networks, xmin, xmax, ymin, ymax



def normalize_dmp_pos(xpos, xmin=-math.pi * 2, xmax=math.pi * 2):

    xpos_clean = []
    for xi in xpos:
        if xi > xmax:
            xpos_clean.append(xmax)
        elif xi < xmin:
            xpos_clean.append(xmin)
        else:
            xpos_clean.append(xi[0])
    return xpos_clean


if __name__ == "__main__":

    K = 50.0
    D = 10.0

    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-g", "--goal", action="store", help="goal angle", type="float")
    parser.add_option("-t", "--train", action="store", help="training", type="string")
    parser.add_option("-l", "--load", action="store", help="load network parameters", type="string")
    parser.add_option("-n", "--num", action="store", help="number of experts", type="int")

    parser.add_option("-r", "--rate", action="store", help="learning rate", type="float")
    parser.add_option("-d", "--decay", action="store", help="decay rate", type="float")


    (options, args) = parser.parse_args()

    target_theta = 0.0 if options.goal is None else options.goal

    y = -200 * math.sin(target_theta)
    x = 200 * math.cos(target_theta)

    pygame.init()
    display = pygame.display.set_mode((width, height))
    fpsClock = pygame.time.Clock()
    world = ResetWorld(origin, width, height, x, y)


    if options.load is not None:
        f = open(options.load, "r")
        data = pickle.load(f)
        networks = data[0]
        xmin = data[1]
        xmax = data[2]
        ymin = data[3]
        ymax = data[4]

    else:
        networks, xmin, xmax, ymin, ymax = trainNetworks(options)
        f = open("network_parameters.pkl", "w")
        pickle.dump((networks, xmin, xmax, ymin, ymax), f)


    parameters = []
    feat = np.zeros( (1, 3) )
    feat[0,0] = (x - xmin[0]) / (xmax[0] - xmin[0])
    feat[0,1] = (y - xmin[1]) / (xmax[1] - xmin[1])
    feat[0,2] = target_theta

    all_params=[]
    for i, network in enumerate(networks):
        new_feat = network.transform_features(feat)
        prediction, expertsPrediction = network.computeMixtureOutput(new_feat)
        prediction = (prediction * (ymax[i] - ymin[i])) + ymin[i]
        parameters.append(prediction)


    dmp1 = DMP(basis, K, D, world.arm.joint1.angle, parameters[0])
    dmp2 = DMP(basis, K, D, world.arm.joint2.angle, parameters[1])
    dmp3 = DMP(basis, K, D, world.arm.joint3.angle, parameters[2])

    all_pos  = list()
    count = 3
    for i in range(basis):
        dmp1.weights[i] = parameters[count]
        dmp2.weights[i] = parameters[count+basis]
        dmp3.weights[i] = parameters[count+(2*basis)]
        count += 1


    x1, x1dot, x1ddot, t1 = dmp1.run_dmp(tau, dmp_dt, dmp1.start, dmp1.goal)
    x2, x2dot, x2ddot, t2 = dmp2.run_dmp(tau, dmp_dt, dmp2.start, dmp2.goal)
    x3, x3dot, x3ddot, t3 = dmp3.run_dmp(tau, dmp_dt, dmp3.start, dmp3.goal)

    x1 = normalize_dmp_pos(x1)
    x2 = normalize_dmp_pos(x2)
    x3 = normalize_dmp_pos(x3, 0, math.pi)

    plt.plot(t1, x1, "b")
    plt.show()

    plt.plot(t2, x2, "r")
    plt.show()

    plt.plot(t3, x3, "g")
    plt.show()

    RunSimulation(world, x1, x2, x3, display, height, x, y, dt, fpsClock, FPS)







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
tool_segments = 4


width = 1000
height = 1000

FPS = 60
dt = 1.0 / FPS
origin = (width / 2+150, (height / 4)*3 - 400)
dmp_dt = 0.1
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

    if len(training_x.shape) == 1:
        training_x = training_x.reshape( (training_x.shape[0], 1) )
    if len(training_y.shape) == 1:
        training_y = training_y.reshape( (training_y.shape[0], 1) )

    return training_x, training_y, np.asarray(xmin), np.asarray(xmax), np.asarray(ymin), np.asarray(ymax)

def trainNetworks(options):
    experts = 1 if options.num is None else options.num

    learningRate = 0.01 if options.rate is None else options.rate
    decay = 0.98 if options.decay is None else options.decay

    training_x, training_y, xmin, xmax, ymin, ymax = generateTrainingData(options.train)

    networks = []

    indexes = np.random.permutation(training_x.shape[0])
    training_x = training_x[indexes]
    training_y = training_y[indexes]


    mixExperts = MixtureOfExperts(experts, "em", "coop", training_x, training_y, poly_degree=1, feat_type="polynomial")
    mixExperts.learningRate = learningRate
    mixExperts.decay = decay


    test_x = training_x[:training_x.shape[0]/4]
    train_x = training_x[training_x.shape[0]/4:]
    test_y = training_y[:training_y.shape[0]/4]
    train_y = training_y[training_y.shape[0]/4:]

    mixExperts.trainNetwork(train_x, train_y, test_x, test_y, 50)

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


def generate_tool_parameters(params, num_basis, num_segments):
    pi2 = math.pi * 2.0
    tool_parameters = []
    for i in range(num_segments):
        segment_length = params[num_basis*4+(i*2)] if params[num_basis*4+(i*2)] > 10 else 10
        segment_angle = params[num_basis*4+(i*2)+1] % pi2
        tool_parameters.append( (segment_length, segment_angle) )
    return tool_parameters


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
    parser.add_option("-m", "--moe", action="store", help="moe file name", type="string")


    (options, args) = parser.parse_args()

    target_theta = 0.0 if options.goal is None else options.goal

    y = 200 * math.sin(target_theta)
    x = 200 * math.cos(target_theta)


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
        f = open(options.moe, "w")
        pickle.dump((networks, xmin, xmax, ymin, ymax), f)


    parameters = []
    feat = np.zeros( (1, 2) )
    feat[0,0] = (x - xmin[0]) / (xmax[0] - xmin[0])
    feat[0,1] = (y - xmin[1]) / (xmax[1] - xmin[1])

    for i, network in enumerate(networks):
        new_feat = network.transform_features(feat)
        prediction, expertsPrediction = network.computeMixtureOutput(new_feat)
        parameters = (prediction * (ymax - ymin)) + ymin



    pygame.init()
    display = pygame.display.set_mode((width, height))
    fpsClock = pygame.time.Clock()
    tool_parameters = generate_tool_parameters(parameters, basis, tool_segments)
    world = ResetWorld(origin, width, height, x, y, tool_parameters)


    dmp1reach = DMP(basis, K, D, world.arm.pivot_position3[0]+world.arm.tool.length, world.domain_object.body.position[0])
    dmp2reach = DMP(basis, K, D, world.arm.pivot_position3[1], world.domain_object.body.position[1])

    all_pos  = list()
    for i in range(basis):
        dmp1reach.weights[i] = parameters[0,i]
        dmp2reach.weights[i] = parameters[0,i+basis]

    x1, x1dot, x1ddot, t1 = dmp1reach.run_dmp(tau, dmp_dt, dmp1reach.start, dmp1reach.goal)
    x2, x2dot, x2ddot, t2 = dmp2reach.run_dmp(tau, dmp_dt, dmp2reach.start, dmp2reach.goal)
    l1, l2 = [], []
    # x1 = normalize_dmp_pos(x1)
    for i in range(len(x1)):
        l1.append( x1[i] )
    all_pos.append( l1 )
    # all_pos[0].append(dmp1reach.goal)

    # x2 = normalize_dmp_pos(x2)
    for i in range(len(x2)):
        l2.append( x2[i] )
    all_pos.append( l2 )
    # all_pos[1].append(dmp2reach.goal)

    dmp1push = DMP(basis, K, D, world.domain_object.body.position[0], world.domain_object.target_position[0])
    dmp2push = DMP(basis, K, D, world.domain_object.body.position[1], world.domain_object.target_position[1])
    for i in range(basis):
        dmp1push.weights[i] = parameters[0,i+(2*basis)]
        dmp2push.weights[i] = parameters[0,i+(3*basis)]


    x1, x1dot, x1ddot, t1 = dmp1push.run_dmp(tau, dmp_dt, dmp1push.start, dmp1push.goal)
    x2, x2dot, x2ddot, t2 = dmp2push.run_dmp(tau, dmp_dt, dmp2push.start, dmp2push.goal)

    # x1 = normalize_dmp_pos(x1)
    for i in range(len(x1)):
        all_pos[0].append( x1[i] )
    all_pos[0].append(dmp1push.goal)

    # x2 = normalize_dmp_pos(x2)
    for i in range(len(x2)):
        all_pos[1].append( x2[i] )
    all_pos[1].append(dmp2push.goal)

    RunSimulation(world, all_pos[0], all_pos[1], display, height, x, y, dt, fpsClock, FPS)







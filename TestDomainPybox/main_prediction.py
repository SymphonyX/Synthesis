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
basis = 3
tool_segments = 4
num_reach_dmps = 2
num_push_dmps = 2
total_dmps = num_push_dmps + num_reach_dmps


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

    return networks, xmin, xmax, ymin, ymax


def generate_tool_parameters(params, num_basis, num_segments):
    pi2 = math.pi * 2.0
    tool_parameters = []
    for i in range(num_segments):
        segment_length = params[(num_basis*total_dmps*2)+(i*2)] if params[(num_basis*total_dmps*2)+(i*2)] > 50 else 50
        if segment_length > 200:
            segment_length = 200
        segment_angle = 0 if i == 0 else params[(num_basis*total_dmps*2)+(i*2)+1] % pi2
        tool_parameters.append( (segment_length, segment_angle) )

    # tool_parameters.append( (100.0, 0.0) )
    # tool_parameters.append( (50.0, math.pi / 2.0) )

    return tool_parameters

def generate_starts_and_goals_lists(goal_params, world):

    goals, starts = [], []
    for i in range(total_dmps):
        if i < num_reach_dmps-1:
            goals.append( goal_params[2*i] )
            goals.append( goal_params[2*i+1] )
        elif i == num_reach_dmps-1:
            goals.append( world.domain_object.body.position[0] )
            goals.append( world.domain_object.body.position[1] )
        elif i < total_dmps-1: #Skip the goal of reaching, that's fixed
            goals.append( goal_params[(2*(i-1))] )
            goals.append( goal_params[(2*(i-1)+1)] )
        else:
            goals.append( world.domain_object.target_position[0] )
            goals.append( world.domain_object.target_position[1] )

        if i == 0:
            starts.append( world.arm.pivot_position3[0]+world.arm.tool.length )
            starts.append( world.arm.pivot_position3[1] )
        else:
            starts.append( goals[-4] )
            starts.append( goals[-3] )

    return starts, goals


def generate_dmps_from_parameters(params, num_basis, starts, goals, K, D):

    dmps_list = []
    for i in range(len(starts)):
        dmp = DMP(num_basis, K, D, starts[i], goals[i])
        for j in range(num_basis):
            dmp.weights[j] = params[j+(num_basis*i)]
        dmps_list.append(dmp)
    return dmps_list


def positions_from_dmps(dmp_list):
    all_pos = [ [], [] ]
    for i, dmp in enumerate(dmp_list):
        xpos, xdot, xddot, times = dmp.run_dmp(tau, dmp_dt, dmp.start, dmp.goal)
        for j in range( len(xpos) ):
            all_pos[ (i % 2) ].append( xpos[j] )

    all_pos[0].append( dmp_list[-2].goal ) #Append x target position
    all_pos[1].append( dmp_list[-1].goal ) #Append y target position

    return all_pos


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
        prediction = (prediction * (ymax[i] - ymin[i])) + ymin[i]
        parameters.append(prediction)



    pygame.init()
    display = pygame.display.set_mode((width, height))
    fpsClock = pygame.time.Clock()


    tool_parameters = generate_tool_parameters(parameters, basis, tool_segments)
    world = ResetWorld(origin, width, height, x, y, tool_parameters)

    goals_prediction = parameters[-(total_dmps-2)*2:]
    starts, goals = generate_starts_and_goals_lists(goals_prediction, world)

    dmps_list = generate_dmps_from_parameters(parameters, basis, starts, goals, K, D)

    all_pos = positions_from_dmps(dmps_list)

    RunSimulation(world, all_pos[0], all_pos[1], display, height, x, y, dt, fpsClock, FPS)







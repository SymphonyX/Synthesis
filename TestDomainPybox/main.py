import pygame
import sys
sys.path.insert(0, "../")

import matplotlib.pyplot as plt

from dmp import DMP
from scipy import optimize
import numpy as np
import math
from Box2D import *
import pickle
from optparse import OptionParser
from domain import ResetWorld
from domain import SetJointsIteration
from domain import MoveJointsIteration
from domain import RunSimulation
from domain import UndesiredContact

tau = 2.0
basis = 5

width = 1000
height = 1000

FPS = 60
dt = 1.0 / FPS
origin = (width / 2, (height / 4)*3 - 550)
dmp_dt = 0.2
fpsClock = None

best_error = float("inf")
best_params = None
best_distance = float("inf")
target_theta = math.pi



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

def error_func(x):

    world = ResetWorld(origin, width, height, target_theta)
    step = 0

    dmp1 = DMP(basis, K, D, world.arm.joint1.angle, x[0])
    dmp2 = DMP(basis, K, D, world.arm.joint2.angle, x[1])
    dmp3 = DMP(basis, K, D, world.arm.joint3.angle, x[2])

    all_pos  = list()
    count = 3
    for i in range(basis):
        dmp1.weights[i] = x[count]
        dmp2.weights[i] = x[count+basis]
        dmp3.weights[i] = x[count+(2*basis)]
        count += 1

    xpos, xdot, xddot, times = dmp1.run_dmp(tau, dmp_dt, dmp1.start, dmp1.goal)
    xpos_clean = normalize_dmp_pos(xpos)
    all_pos.append( xpos_clean )

    xpos, xdot, xddot, times = dmp2.run_dmp(tau, dmp_dt, dmp2.start, dmp2.goal)
    xpos_clean = normalize_dmp_pos(xpos)
    all_pos.append( xpos_clean )

    xpos, xdot, xddot, times = dmp3.run_dmp(tau, dmp_dt, dmp3.start, dmp3.goal)
    xpos_clean = normalize_dmp_pos(xpos, 0, math.pi)
    all_pos.append( xpos_clean )


    sum_distances = 0
    for i in range(len(all_pos[0])-1):
        sum_distances += math.fabs(all_pos[0][i+1] - all_pos[0][i])
        sum_distances += math.fabs(all_pos[1][i+1] - all_pos[1][i])
        sum_distances += math.fabs(all_pos[2][i+1] - all_pos[2][i])

    thetas_reached = True
    tool_distance = 0.0
    traveled_distance = 0.0
    penalty = 0
    total_error = 0.0
    total_steps = 0
    while step < len(all_pos[0]) or thetas_reached == False:
        penalty = 0
        prev_pos = world.domain_object.body.position.copy()
        if thetas_reached == True:
            SetJointsIteration(all_pos[0][step], all_pos[1][step], all_pos[2][step], world)
            step += 1
            thetas_reached = False
            contact = 0
        else:
            thetas_reached = MoveJointsIteration(world.arm.joint1, world.arm.joint2, world.arm.joint3)
            tool_distance += math.sqrt( (world.domain_object.body.position[0] - world.arm.tool.body2.position[0])**2 \
                                 + (world.domain_object.body.position[1] - world.arm.tool.body2.position[1])**2 ) #TODO this is only checking against the center


        world.Step(dt, 20, 20)
        world.ClearForces()
        traveled_distance += math.sqrt( (prev_pos[0] - world.domain_object.body.position[0])**2 + (prev_pos[1] - world.domain_object.body.position[1])**2)

        object_contact = False
        for edge in world.domain_object.body.contacts:
            data1 = edge.contact.fixtureA.body.userData
            data2 = edge.contact.fixtureB.body.userData
            if data1 == "tool1" or data1 == "tool2" or data2 == "tool1" or data2 == "tool2":
                object_contact = True

        if object_contact == False:
            world.domain_object.body.angularVelocity = 0.0
            world.domain_object.body.linearVelocity = b2Vec2(0,0)

        if UndesiredContact(world):
            contact += 1
            penalty += 1000
            if contact == 1000:
                penalty = 100000
                break

        if total_steps > 50000:
            total_steps = 1 #make error large so we don't pick it
            print "Escape..."
            break


        total_error += math.sqrt( (world.domain_object.target_position[0] - world.domain_object.body.position[0])**2 + (world.domain_object.target_position[1] - (height - world.domain_object.body.position[1]))**2)
        total_steps += 1

    #total_error =  (1000*error) + tool_distance  + penalty + 0.1*traveled_distance + 0.1*sum_distances#( 5 * (np.linalg.norm(all_pos[0]) + np.linalg.norm(all_pos[1])))
    cost = (1000 * (total_error/total_steps)) + ( 100.0 * np.linalg.norm(x))
    error = math.sqrt( (world.domain_object.target_position[0] - world.domain_object.body.position[0])**2 \
                                   + (world.domain_object.target_position[1] - (height - world.domain_object.body.position[1]))**2)
    global best_error
    global best_params
    global best_distance

    if cost <= best_error:
        best_error = cost
        best_params = x
        best_distance = error

    print "\nAvg Error: ", (total_error/total_steps)
    print "Error: ", error
    print "Trajectory length: ", sum_distances
    print "Cost: ", cost

    return cost


def diff_demonstration(demonstration, time):
    velocities = np.zeros( (len(demonstration), 1) )
    accelerations = np.zeros( (len(demonstration), 1) )

    times = np.linspace(0, time, num=len(demonstration))

    for i in range(1,len(demonstration)):
        dx = demonstration[i] - demonstration[i-1]
        dt = times[i] - times[i-1]

        velocities[i] = dx / dt
        accelerations[i] = (velocities[i] - velocities[i-1]) / dt

    velocities[0] = velocities[1] - (velocities[2] - velocities[1])
    accelerations[0] = accelerations[1] - (accelerations[2] - accelerations[1])

    return demonstration, velocities, accelerations, times





if __name__ == '__main__':

    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-s", "--save", action="store", help="save param file", type="string")
    parser.add_option("-l", "--load", action="store", help="load param file", type="string")
    parser.add_option("-t", "--target", action="store", help="target angle", type="float")
    parser.add_option("-p", "--params", action="store", help="parameters initial values", type="string")


    (options, args) = parser.parse_args()
    target_theta = 0.0 if options.target is None else options.target

    K = 50.0
    D = 10.0


    if options.load is not None:
        print "Loading params from file"
        param_file = open(options.load, "r")
        result = pickle.load(param_file)
    else:
        params = None
        outer_iter = 1 if options.params is not None else 20
        for j in range(outer_iter):
            iterations = 5
            if options.params is not None:
                if params is None:
                    params_file = open(options.params, "r")
                    params = pickle.load(params_file)
                epsilons = np.zeros( (basis*3+3) )
                epsilons[:3] = 10.5
                epsilons[3:] = 50.5
                iterations = 10
            elif options.params is None:
                params = np.random.uniform( -10, 10, (basis*3+3, 1) )
                params[:3] = np.random.uniform(-2*math.pi, 2*math.pi)
                epsilons = np.zeros( (basis*3+3) )
                epsilons[:3] = 0.05
                epsilons[3:] = 0.5

            for i in range(iterations):

                result = optimize.fmin_bfgs(f=error_func, x0=[ params ], epsilon=epsilons)
                epsilons[:3] = epsilons[:3] / 10.0
                epsilons[3:] = epsilons[3:] / 10.0

                params = best_params

        print "Best error: ", best_error
        print "Best params: ", best_params
        print "Best distances: ", best_distance, "\n\n"

        result = best_params

        filename = "params.pkl" if options.save is None else options.save
        param_file = open(filename, "w")
        pickle.dump(result, param_file)
        error_filename = filename.split(".")[0] + "_error.txt"
        error_file = open(error_filename, "w")
        error_file.write("Error: " + str(best_distance))
        error_file.write("\nCost: " + str(best_error))
        error_file.close()
        param_file.close()

    pygame.init()
    display = pygame.display.set_mode((width, height))
    fpsClock = pygame.time.Clock()
    world = ResetWorld(origin, width, height, target_theta)


    dmp1 = DMP(basis, K, D, world.arm.joint1.angle, result[0])
    dmp2 = DMP(basis, K, D, world.arm.joint2.angle, result[1])
    dmp3 = DMP(basis, K, D, world.arm.joint3.angle, result[2])


    all_pos  = list()
    count = 3
    for i in range(basis):
        dmp1.weights[i] = result[count]
        dmp2.weights[i] = result[count+basis]
        dmp3.weights[i] = result[count+(2*basis)]
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

    RunSimulation(world, x1, x2, x3, display, height, target_theta, dt, fpsClock, FPS)

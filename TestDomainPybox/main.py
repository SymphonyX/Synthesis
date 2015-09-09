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

tau = 2.0
basis = 5

width = 1000
height = 1000

FPS = 60
dt = 1.0 / FPS
origin = (width / 2+150, (height / 4)*3 - 400)
dmp_dt = 0.1
fpsClock = None

best_error = float("inf")
best_params = None
best_distance = float("inf")
target_x = 0.0
target_y = 0.0



def normalize_dmp_pos(xpos, xmin=0, xmax=800):

    xpos_clean = []
    for xi in xpos:
        if xi > xmax:
            xpos_clean.append(xmax)
        elif xi < xmin:
            xpos_clean.append(xmin)
        else:
            xpos_clean.append(int(xi[0]))
    return xpos_clean

def error_func(x):

    world = ResetWorld(origin, width, height, target_x, target_y)
    step = 0

    dmp1reach = DMP(basis, K, D, world.arm.pivot_position3[0]+world.arm.tool.length, world.domain_object.body.position[0])
    dmp2reach = DMP(basis, K, D, world.arm.pivot_position3[1], world.domain_object.body.position[1])


    all_pos  = list()
    for i in range(basis):
        dmp1reach.weights[i] = x[i]
        dmp2reach.weights[i] = x[basis+i]

    xpos1, xdot1, xddot1, times = dmp1reach.run_dmp(tau, dmp_dt, dmp1reach.start, dmp1reach.goal)
    l1,l2 = [], []

    for i in range(len(xpos1)):
        l1.append( xpos1[i] )
    all_pos.append(l1)

    xpos2, xdot2, xddot2, times = dmp2reach.run_dmp(tau, dmp_dt, dmp2reach.start, dmp2reach.goal)
    for i in range(len(xpos2)):
        l2.append( xpos2[i] )
    all_pos.append(l2)

    dmp1push = DMP(basis, K, D, world.domain_object.body.position[0], world.domain_object.target_position[0])
    dmp2push = DMP(basis, K, D, world.domain_object.body.position[1], world.domain_object.target_position[1])

    for i in range(basis):
        dmp1push.weights[i] = x[(2*basis)+i]
        dmp2push.weights[i] = x[(3*basis)+i]


    xpos1, xdot1, xddot1, times = dmp1push.run_dmp(tau, dmp_dt, dmp1push.start, dmp1push.goal)
    for i in range(len(xpos1)):
        all_pos[0].append( xpos1[i] )
    all_pos[0].append(dmp1push.goal)

    xpos2, xdot2, xddot2, times = dmp2push.run_dmp(tau, dmp_dt, dmp2push.start, dmp2push.goal)
    for i in range(len(xpos2)):
        all_pos[1].append( xpos2[i] )
    all_pos[1].append(dmp2push.goal)

    sum_distances = 0

    thetas_reached = True
    total_error = 0.0
    total_steps = 0
    pd_step = 0
    while step < len(all_pos[0]) or thetas_reached == False:
        penalty = 0

        prev_pos = world.arm.end_effector_position()

        if thetas_reached == True:
            theta1, theta2, theta3 = world.arm.inverse_kinematics_ccd(all_pos[0][step], all_pos[1][step])
            SetJointsIteration(theta1, theta2, theta3, world)
            step += 1
            #print "Step %d/%d" %(step, len(all_pos[0]))
            thetas_reached = False
            contact = 0
            pd_step = 0
        else:
            thetas_reached = MoveJointsIteration(world.arm.joint1, world.arm.joint2, world.arm.joint3)

        pd_step += 1
        world.Step(dt, 40, 40)
        world.ClearForces()
        if pd_step == 1000:
            print "Escaping..."
            penalty = 10000000
            break


        new_pos = world.arm.end_effector_position()
        sum_distances += math.sqrt( (prev_pos[0] - new_pos[0])**2 + (prev_pos[1] - new_pos[1])**2 )

        object_contact = False
        for edge in world.domain_object.body.contacts:
            data1 = edge.contact.fixtureA.body.userData
            data2 = edge.contact.fixtureB.body.userData
            if data1 == "tool1" or data1 == "tool2" or data2 == "tool1" or data2 == "tool2":
                object_contact = True

        if object_contact == False:
            world.domain_object.body.angularVelocity = 0.0
            world.domain_object.body.linearVelocity = b2Vec2(0,0)


        current_error = math.sqrt( (world.domain_object.target_position[0] - world.domain_object.body.position[0])**2 + (world.domain_object.target_position[1] - world.domain_object.body.position[1])**2)
        total_error += current_error
        total_steps += 1


    error = math.sqrt( (world.domain_object.target_position[0] - world.domain_object.body.position[0])**2 \
                                   + (world.domain_object.target_position[1] -  world.domain_object.body.position[1])**2)
    cost = (1000 * error) + np.linalg.norm(x) #+ (10 * sum_distances) #(10 * (total_steps - goal_reach_step))#

    global best_error
    global best_params
    global best_distance

    if cost <= best_error:
        best_error = cost
        best_params = x
        best_distance = error

    #print "\nAvg Error: ", (total_error/total_steps)
    print "\nError: ", error
    print "X: ", target_x, " Y: ", target_y
    print "Best Error: ", best_distance
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
    parser.add_option("-x", "--xpos", action="store", help="target x", type="float")
    parser.add_option("-y", "--ypos", action="store", help="target y", type="float")
    parser.add_option("-t", "--theta", action="store", help="target theta", type="float")
    parser.add_option("-p", "--params", action="store", help="parameters initial values", type="string")


    (options, args) = parser.parse_args()
    if options.theta is not None:
        target_x = 200 * math.cos(options.theta)
        target_y = 200 * math.sin(options.theta)
    else:
        target_x = 200.0 if options.xpos is None else options.xpos
        target_y = 0.0 if options.ypos is None else options.ypos

    K = 50.0
    D = 10.0


    if options.load is not None:
        print "Loading params from file"
        param_file = open(options.load, "r")
        result = pickle.load(param_file)
    else:
        params = None
        outer_iter = 1 if options.params is not None else 5
        for j in range(outer_iter):
            iterations = 3
            if options.params is not None:
                if params is None:
                    params_file = open(options.params, "r")
                    params = pickle.load(params_file)
                epsilons = np.zeros( (basis*4) )
                epsilons[:] = 10.5
                iterations = 5
            elif options.params is None:
                params = np.random.uniform( -40, 40, (basis*4, 1) )
                epsilons = np.zeros( (basis*4) )
                epsilons[:] = 10.0

            for i in range(iterations):
                status_file = open("status.txt", "w")
                status_file.write("Theta: " + str(options.theta) )
                status_file.write("Outer: " + str(j) + " Inner: " + str(i))
                status_file.close()

                result = optimize.fmin_bfgs(f=error_func, x0=[ params ], epsilon=epsilons, maxiter=10)
                epsilons[:] = epsilons[:] / 10.0

                #params = best_params

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
    world = ResetWorld(origin, width, height, target_x, target_y)


    dmp1reach = DMP(basis, K, D, world.arm.pivot_position3[0]+world.arm.tool.length, world.domain_object.body.position[0])
    dmp2reach = DMP(basis, K, D, world.arm.pivot_position3[1], world.domain_object.body.position[1])


    all_pos  = list()
    for i in range(basis):
        dmp1reach.weights[i] = result[i]
        dmp2reach.weights[i] = result[basis+i]

    xpos1, xdot1, xddot1, times = dmp1reach.run_dmp(tau, dmp_dt, dmp1reach.start, dmp1reach.goal)
    l1,l2 = [], []

    for i in range(len(xpos1)):
        l1.append( xpos1[i] )
    all_pos.append(l1)

    xpos2, xdot2, xddot2, times = dmp2reach.run_dmp(tau, dmp_dt, dmp2reach.start, dmp2reach.goal)
    for i in range(len(xpos2)):
        l2.append( xpos2[i] )
    all_pos.append(l2)

    dmp1push = DMP(basis, K, D, world.domain_object.body.position[0], world.domain_object.target_position[0])
    dmp2push = DMP(basis, K, D, world.domain_object.body.position[1], world.domain_object.target_position[1])

    for i in range(basis):
        dmp1push.weights[i] = result[(2*basis)+i]
        dmp2push.weights[i] = result[(3*basis)+i]


    xpos1, xdot1, xddot1, times = dmp1push.run_dmp(tau, dmp_dt, dmp1push.start, dmp1push.goal)
    for i in range(len(xpos1)):
        all_pos[0].append( xpos1[i] )
    all_pos[0].append(dmp1push.goal)

    xpos2, xdot2, xddot2, times = dmp2push.run_dmp(tau, dmp_dt, dmp2push.start, dmp2push.goal)
    for i in range(len(xpos2)):
        all_pos[1].append( xpos2[i] )
    all_pos[1].append(dmp2push.goal)

    # steps = np.linspace(0, len(all_pos[0]), num=len(all_pos[0]))
    # plt.plot(steps, all_pos[0], label="x-position")
    # plt.plot(steps, all_pos[1], label="y-position")
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #       ncol=2, mode="expand", borderaxespad=0.)
    # plt.show()

    RunSimulation(world, all_pos[0], all_pos[1], display, height, target_x, target_y, dt, fpsClock, FPS)

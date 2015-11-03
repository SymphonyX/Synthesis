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
import math

tau = 2.0
basis = 5
tool_segments = 2

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


num_reach_dmps = 1
num_push_dmps = 1
total_dmps = num_push_dmps + num_reach_dmps

def generate_dmps_from_parameters(params, num_basis, starts, goals, K, D):

    dmps_list = []
    for i in range(len(starts)):
        dmp = DMP(num_basis, K, D, starts[i], goals[i])
        for j in range(num_basis):
            dmp.weights[j] = params[j+(num_basis*i)]
        dmps_list.append(dmp)
    return dmps_list


def generate_tool_parameters(params, num_basis, num_segments):
    pi2 = math.pi * 2.0
    tool_parameters = []
    # for i in range(num_segments):
    #     segment_length = params[(num_basis*total_dmps*2)+(i*2)] if params[(num_basis*total_dmps*2)+(i*2)] > 50 else 50
    #     if segment_length > 200:
    #         segment_length = 200
    #     segment_angle = 0 if i == 0 else params[(num_basis*total_dmps*2)+(i*2)+1] % pi2
    #     tool_parameters.append( (segment_length, segment_angle) )

    tool_parameters.append( (100.0, 0.0) )
    tool_parameters.append( (50.0, math.pi / 2.0) )
    
    return tool_parameters


def generate_starts_and_goals_lists(params, world):
    goals, starts = [], []
    for i in range(total_dmps):
        if i < num_reach_dmps-1:
            start_index = (basis*total_dmps*2)+(tool_segments*2)
            goals.append( params[start_index + (2*i)] )
            goals.append( params[start_index + (2*i+1)] )
        elif i == num_reach_dmps-1:
            goals.append( world.domain_object.body.position[0] )
            goals.append( world.domain_object.body.position[1] )
        elif i < total_dmps-1: #Skip the goal of reaching, that's fixed
            start_index = (basis*total_dmps*2)+(tool_segments*2)
            goals.append( params[start_index + (2*(i-1))] )
            goals.append( params[start_index + (2*(i-1)+1)] )
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

def error_func(x):

    tool_parameters = generate_tool_parameters(x, basis, tool_segments)
    world = ResetWorld(origin, width, height, target_x, target_y, tool_parameters)
    step = 0

    starts, goals = generate_starts_and_goals_lists(x, world)

    dmps_list = generate_dmps_from_parameters(x, basis, starts, goals, K, D)

    all_pos = [ [], [] ]
    for i, dmp in enumerate(dmps_list):
        xpos, xdot, xddot, times = dmp.run_dmp(tau, dmp_dt, dmp.start, dmp.goal)
        for j in range( len(xpos) ):
            all_pos[ (i % 2) ].append( xpos[j] )

    all_pos[0].append( dmps_list[2].goal ) #Append x target position
    all_pos[1].append( dmps_list[3].goal ) #Append y target position

    sum_distances = 0

    thetas_reached = True
    total_error = 0.0
    total_steps = 0
    pd_step = 0
    obstacle_penalty = 0.0

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
        if pd_step == 2000:
            print "Escaping..." 
            penalty = 100
            break


        new_pos = world.arm.end_effector_position()
        sum_distances += math.sqrt( (prev_pos[0] - new_pos[0])**2 + (prev_pos[1] - new_pos[1])**2 )

        object_contact = False
        for edge in world.domain_object.body.contacts:
            data1 = edge.contact.fixtureA.body.userData
            data2 = edge.contact.fixtureB.body.userData
            if data1 == "tool1" or data1 == "tool2" or data2 == "tool1" or data2 == "tool2":
                object_contact = True
            if data1 == "obstacle" or data2 == "obstacle":
                obstacle_penalty += 10

        for body in world.arm.tool.bodies:
            for edge in body.contacts:
                data1 = edge.contact.fixtureA.body.userData
                data2 = edge.contact.fixtureB.body.userData
                if data1 == "obstacle" or data2 == "obstacle":
                    obstacle_penalty += 100


        if object_contact == False:
            world.domain_object.body.angularVelocity = 0.0
            world.domain_object.body.linearVelocity = b2Vec2(0,0)


        current_error = math.sqrt( (world.domain_object.target_position[0] - world.domain_object.body.position[0])**2 + (world.domain_object.target_position[1] - world.domain_object.body.position[1])**2)
        total_error += current_error
        total_steps += 1


    error = math.sqrt( (world.domain_object.target_position[0] - world.domain_object.body.position[0])**2 \
                                   + (world.domain_object.target_position[1] -  world.domain_object.body.position[1])**2)
    tool_length = 0.0
    for i in range(tool_segments):
        tool_length += x[basis*total_dmps*2+i*2]
    cost = (1000 * error)  + obstacle_penalty + (penalty * 0.01 * (len(all_pos[0])-step))#tool_length + np.linalg.norm(x[:basis*4]) #+ (10 * sum_distances) #(10 * (total_steps - goal_reach_step))#

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
    print "Obstacle Penalty: ", obstacle_penalty
    print "Cost: ", cost
    print "Tool params: ", tool_parameters

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
        outer_iter = 1 if options.params is not None else 10
        for j in range(outer_iter):
            iterations = 1
            if options.params is not None:
                if params is None:
                    params_file = open(options.params, "r")
                    params = pickle.load(params_file)
                epsilons = np.zeros( (basis*total_dmps*2 + tool_segments*2 + (num_reach_dmps-1 + num_push_dmps-1)*2) )
                epsilons[:] = 10.5
                for i in range(tool_segments):
                    epsilons[(basis*total_dmps*2)+(2*i)] = 10.0
                    epsilons[(basis*total_dmps*2)+(2*i+1)] = 1.5
                epsilons[(basis*total_dmps*2)+(2*tool_segments):] = 10.0

                iterations = 5

            elif options.params is None:
                params = np.zeros( ((basis*total_dmps*2 )
                                    + tool_segments * 2
                                    + (num_reach_dmps-1 + num_push_dmps-1)*2, 1) )
                epsilons = np.zeros( ((basis*total_dmps*2 )
                                    + tool_segments * 2
                                    + (num_reach_dmps-1 + num_push_dmps-1)*2) )
                epsilons[:] = 10.0

                params[:basis*total_dmps*2] = np.random.uniform(-10, 10)
                for i in range(tool_segments):
                    params[(basis*total_dmps*2)+(2*i)] = np.random.uniform(50, 100)
                    params[(basis*total_dmps*2)+(2*i+1)] = np.random.uniform(-2*math.pi, 2*math.pi)

                    epsilons[(basis*total_dmps*2)+(2*i)] = 10.0
                    epsilons[(basis*total_dmps*2)+(2*i+1)] = 0.2

                params[(basis*total_dmps*2)+(2*tool_segments):] = np.random.uniform(-100, 100)
                epsilons[(basis*total_dmps*2)+(2*tool_segments):] = 10.0



            for i in range(iterations):
                status_file = open("status.txt", "w")
                status_file.write("Theta: " + str(options.theta) )
                status_file.write("Outer: " + str(j) + " Inner: " + str(i))
                status_file.close()

                result = optimize.fmin_bfgs(f=error_func, x0=[ params ], epsilon=epsilons)
                epsilons[:] = epsilons[:] / 10.0

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

    tool_parameters = generate_tool_parameters(result, basis, tool_segments)
    world = ResetWorld(origin, width, height, target_x, target_y, tool_parameters)

    starts, goals = generate_starts_and_goals_lists(result, world)

    dmps_list = generate_dmps_from_parameters(result, basis, starts, goals, K, D)

    all_pos = [ [], [] ]
    for i, dmp in enumerate(dmps_list):
        xpos, xdot, xddot, times = dmp.run_dmp(tau, dmp_dt, dmp.start, dmp.goal)
        for j in range( len(xpos) ):
            all_pos[ (i % 2) ].append( xpos[j] )

    all_pos[0].append( dmps_list[2].goal ) #Append x target position
    all_pos[1].append( dmps_list[3].goal ) #Append y target position

    RunSimulation(world, all_pos[0], all_pos[1], display, height, target_x, target_y, dt, fpsClock, FPS)

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
basis = 3
tool_segments = 4

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

num_reach_dmps = 2
num_push_dmps = 2
total_dmps = num_push_dmps + num_reach_dmps

goals_grid = [ 0 ] * ((num_reach_dmps-1+num_push_dmps-1) * 2)
best_goals = [ 0 ] * ((num_reach_dmps-1+num_push_dmps-1) * 2)

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

def positions_from_dmps(dmp_list):
    all_pos = [ [], [] ]
    for i, dmp in enumerate(dmp_list):
        xpos, xdot, xddot, times = dmp.run_dmp(tau, dmp_dt, dmp.start, dmp.goal)
        for j in range( len(xpos) ):
            all_pos[ (i % 2) ].append( xpos[j] )

    all_pos[0].append( dmp_list[-2].goal ) #Append x target position
    all_pos[1].append( dmp_list[-1].goal ) #Append y target position

    return all_pos

def error_func(x):

    tool_parameters = generate_tool_parameters(x, basis, tool_segments)
    world = ResetWorld(origin, width, height, target_x, target_y, tool_parameters)
    step = 0

    global goals_grid
    starts, goals = generate_starts_and_goals_lists(goals_grid, world)

    dmps_list = generate_dmps_from_parameters(x, basis, starts, goals, K, D)

    print "Goals ", goals

    all_pos = positions_from_dmps(dmps_list)

    sum_distances = 0

    thetas_reached = True
    pd_step = 0
    obstacle_penalty = 0.0
    obj_contact_penalty = 0.0

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
            penalty = 10000
            break


        new_pos = world.arm.end_effector_position()
        sum_distances += math.sqrt( (prev_pos[0] - new_pos[0])**2 + (prev_pos[1] - new_pos[1])**2 )

        object_contact = False
        for edge in world.domain_object.body.contacts:
            data1 = edge.contact.fixtureA.body.userData
            data2 = edge.contact.fixtureB.body.userData
            if data1.startswith("tool") or data2.startswith("tool"):
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
            obj_contact_penalty += 100.0



    error = math.sqrt( (world.domain_object.target_position[0] - world.domain_object.body.position[0])**2 \
                                   + (world.domain_object.target_position[1] -  world.domain_object.body.position[1])**2)

    cost = (1000 * error)  + obstacle_penalty + obj_contact_penalty + penalty + np.linalg.norm(x[:basis*4]) #+ (10 * sum_distances) #(10 * (total_steps - goal_reach_step))#

    global best_error
    global best_params
    global best_distance
    global best_goals

    if error <= best_distance:
        best_error = cost
        best_params = list(x)
        best_distance = error
        best_goals = list(goals_grid)

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


def seed_parameters(options):
    params_file = open(options.params, "r")
    data = pickle.load(params_file)
    params = data[0]
    global goals_grid
    global num_push_dmps
    global num_reach_dmps
    global basis
    global tool_segments
    global total_dmps

    goals_grid = data[1]
    num_push_dmps = data[2]
    num_reach_dmps = data[3]
    total_dmps = num_push_dmps + num_reach_dmps
    tool_segments = data[4]
    basis = data[5]

    epsilons = np.zeros( (basis*total_dmps*2 + tool_segments*2) )
    epsilons[:] = 10.5
    for i in range(tool_segments):
        epsilons[(basis*total_dmps*2)+(2*i)] = 10.0
        epsilons[(basis*total_dmps*2)+(2*i+1)] = 1.5
    epsilons[(basis*total_dmps*2)+(2*tool_segments):] = 10.0

    return params, epsilons


def new_parameters():
    params = np.zeros( ((basis*total_dmps*2 )
                        + tool_segments * 2, 1) )
    epsilons = np.zeros( ((basis*total_dmps*2 )
                        + tool_segments * 2) )
    epsilons[:] = 10.0

    params[:basis*total_dmps*2] = np.random.uniform(-10, 10)
    for i in range(tool_segments):
        params[(basis*total_dmps*2)+(2*i)] = np.random.uniform(50, 100)
        params[(basis*total_dmps*2)+(2*i+1)] = np.random.uniform(-2*math.pi, 2*math.pi)

        epsilons[(basis*total_dmps*2)+(2*i)] = 10.0
        epsilons[(basis*total_dmps*2)+(2*i+1)] = 0.2

    return params, epsilons


if __name__ == '__main__':

    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-s", "--save", action="store", help="save param file", type="string")
    parser.add_option("-l", "--load", action="store", help="load param file", type="string")
    parser.add_option("-x", "--xpos", action="store", help="target x", type="float")
    parser.add_option("-y", "--ypos", action="store", help="target y", type="float")
    parser.add_option("-t", "--theta", action="store", help="target theta", type="float")
    parser.add_option("-p", "--params", action="store", help="parameters initial values", type="string")

    parser.add_option("--basis", action="store", help="number of basis", type="int")
    parser.add_option("--reach", action="store", help="num reach dmps", type="int")
    parser.add_option("--push", action="store", help="num push dmps", type="int")
    parser.add_option("--tool", action="store", help="num tool segments", type="int")


    global goals_grid
    global best_goals

    (options, args) = parser.parse_args()

    if options.basis is not None:
        basis = options.basis
    if options.reach is not None:
        num_reach_dmps = options.reach
    if options.push is not None:
        num_push_dmps = options.push
    if options.tool is not None:
        tool_segments = options.tool
    total_dmps = num_push_dmps + num_reach_dmps


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
        data = pickle.load(param_file)
        result = data[0]
        best_goals = data[1]
        num_push_dmps = data[2]
        num_reach_dmps = data[3]
        total_dmps = num_push_dmps + num_reach_dmps
        tool_segments = data[4]
        basis = data[5]
    else:
        params = None
        iterations = 1
        if options.params is not None:
            params, epsilons = seed_parameters(options)
            iterations = 5

        elif options.params is None:
            params, epsilons = new_parameters()


        max_vals = [1000.0] * len(goals_grid)
        min_vals = [0.0] * len(goals_grid)
        num_steps = 2

        last_index = len(goals_grid)-1
        for k in range(3):
            j = -1
            while True:
                j += 1

                for i in range(iterations):
                    status_file = open("status.txt", "w")
                    status_file.write("Theta: " + str(options.theta) )
                    status_file.write("Outer: " + str(j) + " Inner: " + str(i))
                    status_file.close()

                    result = optimize.fmin_bfgs(f=error_func, x0=[ params ], epsilon=epsilons)
                    epsilons[:] = epsilons[:] / 10.0

                    params = best_params

                index = last_index
                while True:
                    if goals_grid[index] == max_vals[index]:
                        goals_grid[index] = min_vals[index]
                        index -= 1
                    else:
                        step = (max_vals[index] - min_vals[index]) / num_steps
                        goals_grid[index] += step
                        break

                    if index == -1:
                        break

                if options.params is not None or index == -1:
                    break


            if best_error > 30.0:
                for index in range(len(goals_grid)):
                    best_val = best_goals[index]
                    goals_grid[index] = best_val
                    step = (max_vals[index] - min_vals[index]) / num_steps

                    if best_val == min_vals[index]:
                        max_vals[index] = min_vals[index] + step
                    elif best_val == max_vals[index]:
                        min_vals[index] = max_vals[index] - step
                    else:
                        max_vals[index] = best_val + (step / 2.0)
                        min_vals[index] = best_val - (step / 2.0)



        print "Best error: ", best_error
        print "Best params: ", best_params
        print "Best goals: ", best_goals
        print "Best distances: ", best_distance, "\n\n"


        result = best_params

        filename = "params.pkl" if options.save is None else options.save
        param_file = open(filename, "w")
        pickle.dump([result, best_goals, num_push_dmps, num_reach_dmps, tool_segments, basis], param_file)
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

    starts, goals = generate_starts_and_goals_lists(best_goals, world)

    dmps_list = generate_dmps_from_parameters(result, basis, starts, goals, K, D)

    all_pos = positions_from_dmps(dmps_list)

    RunSimulation(world, all_pos[0], all_pos[1], display, height, target_x, target_y, dt, fpsClock, FPS)

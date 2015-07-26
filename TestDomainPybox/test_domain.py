from Arm import Arm
import pygame
import sys
sys.path.insert(0, "../")

import matplotlib.pyplot as plt

from dmp import DMP

from scipy import optimize
import numpy as np
import math
import Box2D
from Box2D import *
from PID import PID
import pickle
from optparse import OptionParser


tau = 2.0
basis = 5

black = (0, 0, 0)
white = (255, 255, 255)
arm_color = (50, 50, 50, 200) # fourth value specifies transparency

width = 1000
height = 1000

FPS = 60
dt = 1.0 / FPS
origin = (width / 2 + 150, (height / 4)*3 - 500)
dmp_dt = 0.2
fpsClock = None

best_error = float("inf")
best_params = None
best_distance = float("inf")
target_theta = math.pi
PD1 = PID(P=100.0, I=0.0, D=15000.0)
PD2 = PID(P=100.0, I=0.0, D=18000.0)


class DomainObject:

    def __init__(self, position, color, radius, world):
        self.position = position
        self.target_radius = 200
        self.target_position = (int(position[0]+(self.target_radius*math.cos(target_theta))),
                                int(position[1]+(self.target_radius*math.sin(target_theta))))
        self.color = color
        self.radius = radius

        fixture=b2FixtureDef(
                        shape=b2PolygonShape(box=(15, 15)),
                        density=0.5,
                        friction=0.1,
                        )

        self.body=world.CreateDynamicBody(
                    position=(position[0], height-position[1]),
                    fixtures=fixture,
                )
        self.body.shape = fixture.shape
        self.body.userData = "target"

    def draw(self):
        vertices=[(self.body.transform*v) for v in self.body.shape.vertices]
        vertices=[(v[0], height-v[1]) for v in vertices]

        pygame.draw.polygon(display, self.color, vertices)
        pygame.draw.circle(display, self.color, self.position, self.target_radius, 10)
        pygame.draw.circle(display, (0, 255, 0, 0), self.target_position, 10)


def convert_angle_ccw(theta):
    angle_ccw = theta % (2 * math.pi)

    return angle_ccw

def convert_angle_cw(theta):
    angle_ccw = convert_angle_ccw(theta)
    angle_cw = -(2 * math.pi) + angle_ccw

    return angle_cw


def undesired_contact(world):
    for edge in world.arm.tool.body1.contacts:
        data1 = edge.contact.fixtureA.body.userData
        data2 = edge.contact.fixtureB.body.userData
        if data1 == "link1" or data1 == "link2" or data2 == "link1" or data2 == "link2":
            return True

    for edge in world.arm.tool.body2.contacts:
        data1 = edge.contact.fixtureA.body.userData
        data2 = edge.contact.fixtureB.body.userData
        if data1 == "link1" or data1 == "link2" or data2 == "link1" or data2 == "link2":
            return True

    for edge in world.domain_object.body.contacts:
        data1 = edge.contact.fixtureA.body.userData
        data2 = edge.contact.fixtureB.body.userData
        if (data1 != "tool2" and data2 != "tool2") or (data1 != "tool1" and data2 != "tool1"):
            return True

    return False

def normalize_dmp_pos(xpos):
    # m = min(xpos)
    # r = max(xpos) - m
    # array = (xpos - m) / r

    # normalized = (array*(2*math.pi))
    # return normalized

    xpos_clean = []
    for xi in xpos:
        if xi > math.pi * 2:
            xpos_clean.append(math.pi * 2)
        elif xi < -math.pi * 2:
            xpos_clean.append(-math.pi * 2)
        else:
            xpos_clean.append(xi[0])
    return xpos_clean

def error_func(x):

    world = reset_world(origin)
    step = 0

    dmp1 = DMP(basis, K, D, world.arm.link1.body.angle, x[0])
    dmp2 = DMP(basis, K, D, world.arm.link2.body.angle, x[1])
    all_pos  = list()
    count = 2
    for i in range(basis):
        dmp1.weights[i] = x[count]
        dmp2.weights[i] = x[count+basis]
        count += 1

    xpos, xdot, xddot, times = dmp1.run_dmp(tau, dmp_dt, dmp1.start, dmp1.goal)
    xpos_clean = normalize_dmp_pos(xpos)
    all_pos.append( xpos_clean )

    xpos, xdot, xddot, times = dmp2.run_dmp(tau, dmp_dt, dmp2.start, dmp2.goal)
    xpos_clean = normalize_dmp_pos(xpos)
    all_pos.append( xpos_clean )


    sum_distances = 0
    for i in range(len(all_pos[0])-1):
        sum_distances += math.fabs(all_pos[0][i+1] - all_pos[0][i])
        sum_distances += math.fabs(all_pos[1][i+1] - all_pos[1][i])

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
            set_joints_iteration(all_pos[0][step], all_pos[1][step], world)
            step += 1
            thetas_reached = False
            contact = 0
        else:
            thetas_reached = move_joints_iteration(world.arm.joint1, world.arm.joint2)
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

        if undesired_contact(world):
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
    cost = (1000 * (total_error/total_steps))# + penalty + ( 10.0 * sum_distances)
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



def update_screen(world):

    world.domain_object.draw()

    bodies = [world.arm.link1.body, world.arm.link2.body, world.arm.tool.body1, world.arm.tool.body2]
    colors = [arm_color, arm_color, (0, 0, 255, 0), (0, 0, 255, 0)]
    for i, body in enumerate(bodies):
        for fixture in body.fixtures:
            shape=body.shape

            vertices=[(body.transform*v) for v in shape.vertices]
            vertices=[(v[0], height-v[1]) for v in vertices]

            pygame.draw.polygon(display, colors[i], vertices)


    # draw circles at joints for pretty
    world.arm.set_pivot_positions()
    pivot1 = (int(world.arm.pivot_position1[0]), int(world.arm.pivot_position1[1]))
    pivot2 = (int(world.arm.pivot_position2[0]), int(world.arm.pivot_position2[1]))
    pygame.draw.circle(display, black, (pivot1[0], height-pivot1[1]), 30)
    pygame.draw.circle(display, arm_color, (pivot1[0], height-pivot1[1]), 12)
    pygame.draw.circle(display, black, (pivot2[0], height-pivot2[1]), 20)
    pygame.draw.circle(display, arm_color, (pivot2[0], height-pivot2[1]), 7)


def reset_world(arm_origin):
    world = b2World(gravity=(0,0), doSleep=True)
    world.domain_object = DomainObject(position=(width/2, height/3), color=(255,0,0), radius=15, world=world)
    world.arm = Arm(arm_origin[0], arm_origin[1], 300, 300)
    world.arm.createBodies(world)
    return world


def set_joints_iteration(theta1, theta2, world):
    theta1 = convert_angle_ccw( round(theta1, 3) )
    theta2 = convert_angle_ccw( round(theta2, 3) )

    angle1 = round(world.arm.joint1.angle, 3)
    angle2 = round(world.arm.joint2.angle, 3)

    if math.fabs(angle1 - theta1) > math.pi:
        theta1 = convert_angle_cw(theta1)

    if math.fabs(angle2 - theta2) > math.pi:
        theta2 = convert_angle_cw(theta2)

    if theta2 < -math.pi / 1.5:
        theta2 = -math.pi / 1.5
    elif theta2 > math.pi / 1.5:
        theta2 = math.pi / 1.5

    PD1.setPoint(theta1)
    PD2.setPoint(theta2)

def move_joints_iteration(joint1, joint2, printing=False):
    speed1 = PD1.update(joint1.angle) * 1000
    joint1.motorSpeed = speed1

    speed2 = PD2.update(joint2.angle) * 1000
    joint2.motorSpeed = speed2


    error1 = PD1.getError()
    error2 = PD2.getError()

    if printing == True:
        print "Goal: ", PD1.set_point, PD2.set_point
        print "Thetas: ", joint1.angle, joint2.angle
        print "Errors: ", error1, error2
        print "Speeds: ", speed1, speed2
        print "\n\n"

    if math.fabs(error1) < 0.2 and math.fabs(error2) < 0.2:
        return True
    return False


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
        outer_iter = 1 if options.params is not None else 10
        for j in range(outer_iter):
            iterations = 5
            if options.params is not None:
                if params is None:
                    params_file = open(options.params, "r")
                    params = pickle.load(params_file)
                epsilons = np.zeros( (basis*2+2) )
                epsilons[:2] = 0.01
                epsilons[2:] = 0.1
                iterations = 10
            elif options.params is None:
                params = np.random.uniform( -10, 10, (basis*2+2, 1) )
                params[:2] = np.random.uniform(0, 2*math.pi)
                epsilons = np.zeros( (basis*2+2) )
                epsilons[:2] = 0.01
                epsilons[2:] = 0.1

            for i in range(iterations):

                result = optimize.fmin_bfgs(f=error_func, x0=[ params ], epsilon=epsilons)
                epsilons[:2] = epsilons[:2] / 10.0
                epsilons[2:] = epsilons[2:] / 10.0

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

    world = reset_world(origin)


    dmp1 = DMP(basis, K, D, world.arm.link1.body.angle, result[0])
    dmp2 = DMP(basis, K, D, world.arm.link2.body.angle, result[1])
    all_pos  = list()
    count = 2
    for i in range(basis):
        dmp1.weights[i] = result[count]
        dmp2.weights[i] = result[count+basis]
        count += 1


    x1, x1dot, x1ddot, t1 = dmp1.run_dmp(tau, dmp_dt, dmp1.start, dmp1.goal)
    x2, x2dot, x2ddot, t2 = dmp2.run_dmp(tau, dmp_dt, dmp2.start, dmp2.goal)
    x1 = normalize_dmp_pos(x1)
    x2 = normalize_dmp_pos(x2)

    plt.plot(t1, x1, "b")
    plt.show()

    plt.plot(t2, x2, "r")
    plt.show()


    thetas_reached = True
    step = 0
    while step < len(x1) or thetas_reached == False:
        display.fill(white)

        if thetas_reached == True:
            set_joints_iteration(x1[step], x2[step], world)
            step += 1
            thetas_reached = False
            contact = 0
        else:
            thetas_reached = move_joints_iteration(world.arm.joint1, world.arm.joint2, printing=True)

        update_screen(world)

        # check for quit
        for event in pygame.event.get():
            if event.type == pygame.locals.QUIT:
                pygame.quit()
                sys.exit()


        world.Step(dt, 20, 20)
        world.ClearForces()


        error = math.sqrt( (world.domain_object.target_position[0] - world.domain_object.body.position[0])**2 + (world.domain_object.target_position[1] - (height - world.domain_object.body.position[1]))**2)
        print "Step %d/%d" %(step, len(x1))
        print "Error: ", error

        font = pygame.font.SysFont('Arial', 25)
        display.blit(font.render('Goal: pi = ' + str(target_theta), True, (0,0,0)), (200, 100))
        display.blit(font.render("Error: " + str(error), True, (0,0,0)), (200, 150))
        
        pygame.display.flip()

        object_contact = False
        for edge in world.domain_object.body.contacts:
            data1 = edge.contact.fixtureA.body.userData
            data2 = edge.contact.fixtureB.body.userData
            if data1 == "tool1" or data1 == "tool2" or data2 == "tool1" or data2 == "tool2":
                object_contact = True

        if object_contact == False:
            world.domain_object.body.angularVelocity = 0.0
            world.domain_object.body.linearVelocity = b2Vec2(0,0)


        fpsClock.tick(FPS)

        if undesired_contact(world):
            contact += 1
            if contact == 1000:
                break
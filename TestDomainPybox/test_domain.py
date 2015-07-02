from Arm import *
import sys
sys.path.insert(0, "../")

import matplotlib.pyplot as plt

from dmp import DMP

from scipy.optimize import fmin_bfgs
import numpy as np
import math
import Box2D
from Box2D import *
from Box2D.b2 import *
from PID import PID
import pickle

tau = 2.0
basis = 10

black = (0, 0, 0)
white = (255, 255, 255)
arm_color = (50, 50, 50, 200) # fourth value specifies transparency

width = 750
height = 750

FPS = 60
dt = 1.0 / FPS
origin = (width / 2 + 80, (height / 4)*3 - 100)
dmp_dt = 0.02

class DomainObject:

    def __init__(self, position, color, radius, target_radius, world):
        self.position = position
        self.target_position = (position[0]+target_radius, position[1])
        self.target_radius = target_radius
        self.color = color
        self.radius = radius
        self.box2dpos = (position[0], height-position[1])

        fixture=b2FixtureDef(
                        shape=b2PolygonShape(box=(15, 15)),
                        density=0.5,
                        friction=0.1,
                        )

        self.body=world.CreateDynamicBody(
                    position=self.box2dpos,
                    fixtures=fixture,
                )
        self.body.shape = fixture.shape

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



def error_func(x):


    dmp1 = DMP(basis, K, D, x[0], x[1])
    dmp2 = DMP(basis, K, D, x[2], x[3])
    all_pos  = list()
    count = 4
    for i in range(basis):
        dmp1.weights[i] = x[count]
        dmp2.weights[i] = x[count+basis]
        count += 1

    print x

    xpos, xdot, xddot, times = dmp1.run_dmp(tau, dmp_dt, dmp1.start, dmp1.goal)
    all_pos.append( xpos )
    xpos, xdot, xddot, times = dmp2.run_dmp(tau, dmp_dt, dmp2.start, dmp2.goal)
    all_pos.append( xpos )

    sum_distances = 0
    for pos in all_pos:
        for i in range(len(pos)-1):
            sum_distances += math.fabs( (pos[i] % 2*math.pi) - (pos[i+1] % 2 * math.pi) )


    world = reset_world(origin)
    step = 0

    thetas_reached = True
    PD1 = PID(P=100.0, I=0.0, D=1800.0)
    PD2 = PID(P=40.0, I=0.0, D=800.0)

    closest_distance = float("inf")
    while step < len(all_pos[0]) or thetas_reached == False:
        if thetas_reached == True:
            set_joints_iteration(all_pos[0][step], all_pos[1][step], PD1, PD2)
            step += 1
            thetas_reached = False
        else:
            thetas_reached = move_joints_iteration(world.arm.joint1, world.arm.joint2, PD1, PD2)
            distance = math.sqrt( (world.domain_object.body.position[0] - world.arm.link2.body.position[0])**2 \
                                 + (world.domain_object.body.position[1] - world.arm.link2.body.position[1])**2 ) #TODO this is only checking against the center
            if distance < closest_distance:
                closest_distance = distance

        world.Step(dt, 20, 20)
        world.ClearForces()

        if len(world.domain_object.body.contacts) == 0:
            world.domain_object.body.angularVelocity = 0.0
            world.domain_object.body.linearVelocity = b2Vec2(0,0)


    
    error = math.sqrt( (world.domain_object.target_position[0] - world.domain_object.body.position[0])**2 + (world.domain_object.target_position[1] - (height - world.domain_object.body.position[1]))**2)

    print "\nError: ", error
    print "Distances: ", sum_distances
    print "Cost: ", (100*error) + distance + sum_distances
    return (100*error) + distance + sum_distances


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

    bodies = [world.arm.link1.body, world.arm.link2.body]
    for i, body in enumerate(bodies):
        for fixture in body.fixtures:
            shape=body.shape

            vertices=[(body.transform*v) for v in shape.vertices]
            vertices=[(v[0], height-v[1]) for v in vertices]

            pygame.draw.polygon(display, arm_color, vertices)


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
    world.domain_object = DomainObject(position=(width/2, height/3), color=(255,0,0), radius=15, target_radius=200, world=world)
    world.arm = Arm(arm_origin[0], arm_origin[1], 170, 150)
    world.arm.createBodies(world, height)
    return world


def set_joints_iteration(theta1, theta2, PD1, PD2):
    theta1 = convert_angle_ccw( round(theta1, 3) )
    theta2 = convert_angle_ccw( round(theta2, 3) )

    angle1 = round(world.arm.joint1.angle, 3)
    angle2 = round(world.arm.joint2.angle, 3)

    if math.fabs(angle1 - theta1) > math.pi:
        theta1 = convert_angle_cw(theta1)

    if math.fabs(angle2 - theta2) > math.pi:
        theta2 = convert_angle_cw(theta2)

    PD1.setPoint(theta1)
    PD2.setPoint(theta2)

def move_joints_iteration(joint1, joint2, PD1, PD2, printing=False):
    speed1 = PD1.update(joint1.angle) * 10000
    joint1.motorSpeed = speed1

    speed2 = PD2.update(joint2.angle) * 10000
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
    pygame.init()

    display = pygame.display.set_mode((width, height))
    fpsClock = pygame.time.Clock()

    world = reset_world(origin)

    K = 1000.0
    D = 40.0

    if len(sys.argv) > 1:
        print "Loading params from file"
        param_file = open(sys.argv[1], "r")
        result = pickle.load(param_file)
    else:
        params = np.random.uniform( 0, 2*math.pi, (basis*2+4, 1) )
        epsilons = np.zeros( (basis*2+4) )
        epsilons[:4] = 0.00001
        epsilons[4:] = 0.001
        print epsilons.shape
        result = fmin_bfgs(error_func, [ params ], epsilon=epsilons)
        print result

        param_file = open("params.pkl", "w")
        pickle.dump(result, param_file)


    dmp1 = DMP(basis, K, D, result[0], result[1])
    dmp2 = DMP(basis, K, D, result[2], result[3])
    all_pos  = list()
    count = 4
    for i in range(basis):
        dmp1.weights[i] = result[count]
        dmp2.weights[i] = result[count+basis]
        count += 1


    x1, x1dot, x1ddot, t1 = dmp1.run_dmp(tau, dmp_dt, dmp1.start, dmp1.goal)
    plt.plot(t1, x1, "b")
    plt.show()

    x2, x2dot, x2ddot, t2 = dmp2.run_dmp(tau, dmp_dt, dmp2.start, dmp2.goal)
    plt.plot(t2, x2, "b")
    plt.show()



    thetas_reached = True
    step = 0

    PD1 = PID(P=100.0, I=0.0, D=1800.0)
    PD2 = PID(P=40.0, I=0.0, D=800.0)
    while True:    
        display.fill(white)

        if thetas_reached == True:
            set_joints_iteration(x1[step], x2[step], PD1, PD2)
            if step < len(x1)-1:
                step += 1
            thetas_reached = False
        else:
            thetas_reached = move_joints_iteration(world.arm.joint1, world.arm.joint2, PD1, PD2, printing=True)

        update_screen(world)
        
        # check for quit
        for event in pygame.event.get():
            if event.type == pygame.locals.QUIT:
                pygame.quit()
                sys.exit()


        world.Step(dt, 20, 20)
        world.ClearForces()
        pygame.display.flip()

        if len(world.domain_object.body.contacts) == 0:
            world.domain_object.body.angularVelocity = 0.0
            world.domain_object.body.linearVelocity = b2Vec2(0,0)


        fpsClock.tick(FPS)

        error = math.sqrt( (world.domain_object.target_position[0] - world.domain_object.body.position[0])**2 + (world.domain_object.target_position[1] - (height - world.domain_object.body.position[1]))**2)
        print "Step %d/%d" %(step, len(x1))
        print "Error: ", error
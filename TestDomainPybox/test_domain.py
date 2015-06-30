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

dmps = list()
tau = 2.0
basis = 20

black = (0, 0, 0)
white = (255, 255, 255)
arm_color = (50, 50, 50, 200) # fourth value specifies transparency

width = 750
height = 750

FPS = 60
dt = 1.0 / FPS

class DomainObject:

    def __init__(self, position, color, radius, target_radius, world):
        self.position = position
        self.target_position = position
        self.target_radius = target_radius
        self.color = color
        self.radius = radius
        self.box2dpos = (position[0], height-position[1])

        fixture=b2FixtureDef(
                        shape=b2PolygonShape(box=(15, 15)),
                        density=2,
                        friction=50.0,
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
        pygame.draw.circle(display, self.color, self.target_position, self.target_radius, 10)



def convert_angle_ccw(theta):
    angle_ccw = theta % (2 * math.pi)

    return angle_ccw

def convert_angle_cw(theta):
    angle_ccw = convert_angle_ccw(theta)
    angle_cw = -(2 * math.pi) + angle_ccw

    return angle_cw



def error_func(x):
    #x needs to be matrix of shape (num_dmps, num_basis)
    final_pos = list()
    all_pos  = list()
    count = 0
    for i in range(basis):
        for j, dmp in enumerate(dmps):
            dmp.weights[i] = x[count]
            count += 1

    for dmp in dmps:
        xpos, xdot, xddot, times = dmp.run_dmp(tau, 0.01, dmp.start, dmp.goal)
        final_pos.append( xpos[-1] )
        all_pos.append( xpos )
    
    error = 0
    for i in range(len(final_pos)):
        print "Final pos ", final_pos[i]
        print "Goal goal ", dmps[i].goal
        distance = 0
        for j in range(len(all_pos[i])-1):
            distance += math.fabs(all_pos[i][j] - all_pos[i][j+1])

        error += 100 * (final_pos[i] - dmps[i].goal)**2 + distance

    print error
    
    return error


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
    world.arm = Arm(arm_origin[0]+50, arm_origin[1]-100, 150, 120)
    world.arm.createBodies(world, height)
    return world


if __name__ == '__main__':
    pygame.init()

    display = pygame.display.set_mode((width, height))
    fpsClock = pygame.time.Clock()
    origin = (width / 2, (height / 4)*3)

    world = reset_world(origin)


    goalx = 100.0
    goaly = -120.0
    startx = 30.0
    starty = 80.0

    K = 1000.0
    D = 40.0

    xseq = np.linspace(startx, goalx, num=100)
    yseq = np.linspace(starty, goaly, num=100)
    step = 0
    step_change = -1

    thetas1, thetas2 = [], []
    for i in range(len(xseq)):
        theta1, theta2 = world.arm.inverse_kinematics(xseq[i], yseq[i])
        thetas1.append(theta1)
        thetas2.append(theta2)

    demonstration, velocities, accelerations, times = diff_demonstration(thetas1, tau)
    dmp1 = DMP(basis, K, D, demonstration[0], demonstration[-1])
    dmp1.learn_dmp(times, demonstration, velocities, accelerations)

    demonstration, velocities, accelerations, times = diff_demonstration(thetas2, tau)
    dmp2 = DMP(basis, K, D, demonstration[0], demonstration[-1])
    dmp2.learn_dmp(times, demonstration, velocities, accelerations)

    x1, x1dot, x1ddot, t1 = dmp1.run_dmp(tau, 0.01, dmp1.start, dmp1.goal)
    plt.plot(times, thetas1, "r")
    plt.plot(t1, x1, "b")
    plt.show()

    x2, x2dot, x2ddot, t2 = dmp2.run_dmp(tau, 0.01, dmp2.start, dmp2.goal)
    plt.plot(times, thetas2, "r")
    plt.plot(t2, x2, "b")
    plt.show()

    dmps.append( dmp1 )
    dmps.append( dmp2 )


    theta1, theta2 = None, None
    thetas_reached = True
    step = 0

    error1, error2 = 0, 0
    angle1, angle2 = world.arm.joint1.angle, world.arm.joint2.angle
    PD1 = PID(P=5.0, I=0.0, D=1200.0)
    PD2 = PID(P=1.0, I=0.0, D=150.0)
    while True:    
        display.fill(white)

        if thetas_reached == True:
            theta1 = convert_angle_ccw( round(x1[step], 3) )
            theta2 = convert_angle_ccw( round(x2[step], 3) )

            angle1 = round(world.arm.joint1.angle, 3)
            angle2 = round(world.arm.joint2.angle, 3)

            if math.fabs(angle1 - theta1) > math.pi:
                theta1 = convert_angle_cw(theta1)

            if math.fabs(angle2 - theta2) > math.pi:
                theta2 = convert_angle_cw(theta2)

            PD1.setPoint(theta1)
            PD2.setPoint(theta2)
            if step < len(x1)-1:
                step += 1
            thetas_reached = False
        else:
            speed1 = PD1.update(world.arm.joint1.angle)
            world.arm.joint1.motorSpeed = speed1 * 10

            speed2 = PD2.update(world.arm.joint2.angle)
            world.arm.joint2.motorSpeed = speed2 * 10


            error1 = PD1.getError()
            error2 = PD2.getError()

            print "Goal: ", theta1, theta2
            print "Thetas: ", angle1, angle2
            print "Errors: ", error1, error2
            print "Speeds: ", speed1, speed2
            print "\n\n"

            if math.fabs(error1) < 0.2 and math.fabs(error2) < 0.2:
                thetas_reached = True

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


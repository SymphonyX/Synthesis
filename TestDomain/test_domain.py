from Arm import *
import sys
sys.path.insert(0, "../")

import matplotlib.pyplot as plt

from dmp import DMP

from scipy.optimize import fmin_bfgs
import numpy as np
import math

class DomainObject:

    def __init__(self, position, color, radius, target_radius):
        self.position = position
        self.target_position = position
        self.target_radius = target_radius
        self.color = color
        self.radius = radius

    def draw(self):
        pygame.draw.circle(display, self.color, self.position, self.radius)
        pygame.draw.circle(display, self.color, self.target_position, self.target_radius, 10)




dmps = list()
tau = 2.0
basis = 20

black = (0, 0, 0)
white = (255, 255, 255)
arm_color = (50, 50, 50, 200) # fourth value specifies transparency

width = 750
height = 750

domain_object = DomainObject(position=(width/2, height/3), color=(255,0,0), radius=15, target_radius=200)

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


def transform(rect, base, arm_part, rotation):
    rect.center += np.asarray(base)
    rect.center += np.array([np.cos(rotation) * arm_part.offset,
                            -np.sin(rotation) * arm_part.offset])
def transform_lines(rect, base, arm_part, arm):
    rotation = arm.link1.rotation if arm_part is arm.link1 else arm.link1.rotation + arm.link2.rotation
    transform(rect, base, arm_part, rotation)
    rect.center += np.array([-rect.width / 2.0, -rect.height / 2.0])


def update_screen(arm, domain_object):
    ua_image, ua_rect = arm.link1.rotate() 
    fa_image, fa_rect = arm.link2.rotate() 


    joints_x = np.cumsum([0, 
                      arm.link1.scale * np.cos(arm.link1.rotation),
                      arm.link2.scale * np.cos(arm.link2.rotation)]) + origin[0]
    joints_y = np.cumsum([0, 
                      arm.link1.scale * np.sin(arm.link1.rotation),
                      arm.link2.scale * np.sin(arm.link2.rotation)]) * -1 + origin[1]

    joints = [(int(x), int(y)) for x,y in zip(joints_x, joints_y)]

    transform(ua_rect, joints[0], arm.link1, arm.link1.rotation)
    transform(fa_rect, joints[1], arm.link2, arm.link2.rotation)


    # rotate arm lines
    line_ua = pygame.transform.rotozoom(arm.link1.surface, np.degrees(arm.link1.rotation), 1)
    line_fa = pygame.transform.rotozoom(arm.link2.surface, np.degrees(arm.link1.rotation+arm.link2.rotation), 1)
    
    # translate arm lines
    lua_rect = line_ua.get_rect()
    transform_lines(lua_rect, joints[0], arm.link1, arm)

    lfa_rect = line_fa.get_rect()
    transform_lines(lfa_rect, joints[1], arm.link2, arm)

    domain_object.draw()

    display.blit(line_ua, lua_rect)
    display.blit(line_fa, lfa_rect)

    # draw circles at joints for pretty
    pygame.draw.circle(display, black, joints[0], 30)
    pygame.draw.circle(display, arm_color, joints[0], 12)
    pygame.draw.circle(display, black, joints[1], 20)
    pygame.draw.circle(display, arm_color, joints[1], 7)





if __name__ == '__main__':
    pygame.init()

    display = pygame.display.set_mode((width, height))
    fpsClock = pygame.time.Clock()
    origin = (width / 2, (height / 4)*3)


    arm = Arm(origin[0], origin[1], 100, 60)
    
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
        theta1, theta2 = arm.inverse_kinematics(xseq[i], yseq[i])
        thetas1.append(theta1)
        thetas2.append(theta2)

    demonstration, velocities, accelerations, times = diff_demonstration(thetas1, tau)
    dmp1 = DMP(basis, K, D, demonstration[0], demonstration[-1])
    #dmp1.learn_dmp(times, demonstration, velocities, accelerations)

    demonstration, velocities, accelerations, times = diff_demonstration(thetas2, tau)
    dmp2 = DMP(basis, K, D, demonstration[0], demonstration[-1])
    #dmp2.learn_dmp(times, demonstration, velocities, accelerations)


    dmps.append( dmp1 )
    dmps.append( dmp2 )
    
    weights = list()
    for i in range(dmp1.weights.shape[0]):
        w = list( (dmp1.weights[i], dmp2.weights[i]) )
        weights.extend(w)

    result = fmin_bfgs(error_func, [ weights ])


    x1, x1dot, x1ddot, t1 = dmp1.run_dmp(tau, 0.01, dmp1.start, dmp1.goal)
    plt.plot(times, thetas1, "r")
    plt.plot(t1, x1, "b")
    plt.show()

    x2, x2dot, x2ddot, t2 = dmp2.run_dmp(tau, 0.01, dmp2.start, dmp2.goal)
    plt.plot(times, thetas2, "r")
    plt.plot(t2, x2, "b")
    plt.show()


    theta1, theta2 = None, None
    while True:    
        display.fill(white)

        theta1 = round(x1[step], 3)
        theta2 = round(x2[step], 3)
        if step < len(x1)-1:
            step += 1

        arm.move_arm_absolute(theta1, theta2)

        update_screen(arm, domain_object)
        
        # check for quit
        for event in pygame.event.get():
            if event.type == pygame.locals.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.update()
        fpsClock.tick(30)


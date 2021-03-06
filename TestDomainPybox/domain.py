import math
import pygame
import sys
from Box2D import *
from Arm import Arm
from PID import PID

PD1 = PID(P=100.0, I=0.0, D=15000.0)
PD2 = PID(P=100.0, I=0.0, D=18000.0)
PD3 = PID(P=100.0, I=0.0, D=20000.0)

black = (0, 0, 0)
white = (255, 255, 255)
arm_color = (50, 50, 50, 200) # fourth value specifies transparency


class DomainObject:

    def __init__(self, position, color, radius, world, x, y, screen_height):
        self.position = position
        self.target_radius = 200
        self.target_position = (int(position[0]+x),
                                int(position[1]-y))
        self.color = color
        self.radius = radius

        circle = b2CircleShape()
        circle.radius = 15
        circle.pos = (0, 0)
        fixture=b2FixtureDef(
                        shape=circle,
                        density=0.5,
                        friction=0.1,
                        )

        self.body=world.CreateDynamicBody(
                    position=(position[0], screen_height-position[1]),
                    fixtures=fixture,
                )
        self.body.shape = fixture.shape
        self.body.userData = "target"

    def draw(self, display, screen_height):
        # vertices=[(self.body.transform*v) for v in self.body.shape.vertices]
        # vertices=[(v[0], height-v[1]) for v in vertices]
        pygame.draw.circle(display, self.color, (int(self.body.position[0]), screen_height-int(self.body.position[1])), 20)
        pygame.draw.circle(display, self.color, self.position, self.target_radius, 10)
        pygame.draw.circle(display, (0, 255, 0, 0), self.target_position, 10)


def ResetWorld(arm_origin, width, height, xpos, ypos):
    world = b2World(gravity=(0,0), doSleep=True)
    world.domain_object = DomainObject(position=(width/2, height/3), color=(255,0,0), radius=15, world=world, x=xpos, y=ypos, screen_height=height)
    world.arm = Arm(arm_origin[0], arm_origin[1], 250, 200)
    world.arm.createBodies(world)
    return world

def UpdateScreen(world, display, height, arm_color):

    world.domain_object.draw(display, height)

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



def convert_angle_ccw(theta):
    angle_ccw = theta % (2 * math.pi)

    return angle_ccw

def convert_angle_cw(theta):
    angle_ccw = convert_angle_ccw(theta)
    angle_cw = -(2 * math.pi) + angle_ccw

    return angle_cw


def SetJointsIteration(theta1, theta2, theta3, world):
    # theta1 = convert_angle_ccw( round(theta1, 3) )
    # theta2 = convert_angle_ccw( round(theta2, 3) )
    # theta3 = convert_angle_ccw( round(theta2, 3) )

    angle1 = round(world.arm.joint1.angle, 3)
    angle2 = round(world.arm.joint2.angle, 3)
    angle3 = round(world.arm.joint3.angle, 3)

    if math.fabs(angle1 - theta1) > math.pi:
        theta1 = convert_angle_cw(theta1)

    if math.fabs(angle2 - theta2) > math.pi:
        theta2 = convert_angle_cw(theta2)

    if math.fabs(angle3 - theta3) > math.pi:
        theta3 = convert_angle_cw(theta3)

    if theta2 < -math.pi / 1.5:
        theta2 = -math.pi / 1.5
    elif theta2 > math.pi / 1.5:
        theta2 = math.pi / 1.5

    PD1.setPoint(theta1)
    PD2.setPoint(theta2)
    PD3.setPoint(theta3)



def MoveJointsIteration(joint1, joint2, joint3, printing=False):
    speed1 = PD1.update(joint1.angle) * 1000
    joint1.motorSpeed = speed1

    speed2 = PD2.update(joint2.angle) * 1000
    joint2.motorSpeed = speed2

    speed3 = PD3.update(joint3.angle) * 1000
    joint3.motorSpeed = speed3


    error1 = PD1.getError()
    error2 = PD2.getError()
    error3 = PD3.getError()

    if printing == True:
        print "Goal: ", PD1.set_point, PD2.set_point, PD3.set_point
        print "Thetas: ", joint1.angle, joint2.angle, joint3.angle
        print "Errors: ", error1, error2, error3
        print "Speeds: ", speed1, speed2, speed3
        print "\n\n"

    if math.fabs(error1) < 0.2 and math.fabs(error2) < 0.2 and math.fabs(error3) < 0.2:
        return True
    return False



def UndesiredContact(world):
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


def RunSimulation(world, x1, x2 ,x3, display, height, x, y, dt, fpsClock, FPS):

    thetas_reached = True
    step = 0
    pd_step = 0
    while step < len(x1) or thetas_reached == False:
        display.fill(white)

        if thetas_reached == True:
            SetJointsIteration(x1[step], x2[step], x3[step], world)
            step += 1
            thetas_reached = False
            contact = 0
            pd_step = 0
        else:
            thetas_reached = MoveJointsIteration(world.arm.joint1, world.arm.joint2, world.arm.joint3, printing=True)
            if pd_step == 1:
                pd_step = 0
                thetas_reached = True

        pd_step += 1
        UpdateScreen(world, display, height, arm_color)

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
        display.blit(font.render('Goal: (' + str(x) + "," + str(y) + ")", True, (0,0,0)), (200, 100))
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

        if UndesiredContact(world):
            contact += 1
            if contact == 1000:
                break

import math
import pygame
import pygame.locals
import sys
import numpy as np
import Box2D
from Box2D.b2 import *
from Box2D import *

from armpart import ArmPart

from Tool import Tool

class Arm:

    def __init__(self, basex, basey, length1, length2):
        self.bbox_scale = 1.0
        self.nsteps = 50
        self.link1 = ArmPart(length1, scale=1.0)
        self.link2 = ArmPart(length2, scale=1.0)

        self.basex = basex
        self.basey = basey

        self.l2_angle = 2.5

    def createBodies(self, world):
        self.link1.pos = [ (self.basex, self.basey) ]
        self.link2.pos = [ (self.basex+self.link1.length-20, self.basey-10) ]

        vertices1 = [(-self.link1.length/2.0, -self.link1.line_width/2.0), (self.link1.length/2.0, -self.link1.line_width/2.0), (self.link1.length/2.0, self.link1.line_width/2.0), (-self.link1.length/2.0, self.link1.line_width/2.0)]
        vertices2 = [(-self.link2.length/2.0, -self.link2.line_width/2.0), (self.link2.length/2.0, -self.link2.line_width/2.0), (self.link2.length/2.0, self.link2.line_width/2.0), (-self.link2.length/2.0, self.link2.line_width/2.0)]


        rotated_vertices2 = []
        for vertex in vertices2:
            new_vertex = (((vertex[0] - (vertices2[0][0])) * math.cos(self.l2_angle)) - ((vertex[1] - ((vertices2[3][1]-vertices2[0][1])/2)) * math.sin(self.l2_angle)) + (vertices2[0][0]), \
                          ((vertex[0] - (vertices2[0][0])) * math.sin(self.l2_angle)) + ((vertex[1] - ((vertices2[3][1]-vertices2[0][1])/2)) * math.cos(self.l2_angle)) + ((vertices2[3][1]-vertices2[0][1])/2) )

            rotated_vertices2.append( new_vertex )

        self.link1.createBody(world, vertices1, "link1", density=1.5)
        self.link2.createBody(world, rotated_vertices2, "link2", density=1.0)


        self.set_pivot_positions()

        self.pivot1 = world.CreateKinematicBody(position=self.pivot_position1)
        circle=b2CircleShape(pos=self.pivot1.position, radius=1.0)
        self.pivot1.CreateFixture(shape=circle, density=0.0, friction=0.0)
        self.pivot1.userData = "pivot1"

        self.joint1 = world.CreateRevoluteJoint(bodyA=self.pivot1, bodyB=self.link1.body, anchor=self.pivot1.position, enableMotor=True, maxMotorTorque=100000000, motorSpeed=0.0)
        self.joint2 = world.CreateRevoluteJoint(bodyA=self.link1.body, bodyB=self.link2.body, anchor=self.pivot_position2, enableMotor=True, maxMotorTorque=10000000, motorSpeed=0.0, enableLimit=True, lowerAngle=-5.5, upperAngle=0.2)

        self.tool = Tool(self.pivot_position3[0], self.pivot_position3[1], world, 100.0)
        self.joint3 = world.CreateRevoluteJoint(bodyA=self.link2.body, bodyB=self.tool.body1, anchor=self.pivot_position3, enableMotor=True, maxMotorTorque=10000000, motorSpeed=0.0, enableLimit=True, lowerAngle=-math.pi/4.0, upperAngle=(3.0/4.0)*math.pi)



    def set_pivot_positions(self):
        self.pivot_position1 = (self.basex-self.link1.length/2.0, self.basey)

        self.pivot_position2 = (self.pivot_position1[0]+(self.link1.length * math.cos(self.link1.body.angle)), \
                                                         self.pivot_position1[1]+(self.link1.length * math.sin(self.link1.body.angle)))

        self.pivot_position3 = (self.pivot_position2[0]+((self.link2.length-2) * math.cos(self.l2_angle))+15, \
                                                         self.pivot_position2[1]+((self.link2.length-2) * math.sin(self.l2_angle)))


    def move_arm_absolute(self, theta1, theta2):
        self.link1.rotation = theta1
        self.link2.rotation = theta2

        self.link1.pos = [(self.basex, self.basey), \
                        (self.basex + self.link1.length * math.cos(self.link1.rotation), self.basey + self.link1.length * math.sin(self.link1.rotation))
                        ]
        self.link2.pos = [self.link1.pos[1], \
                        (self.link1.pos[1][0] + self.link1.length * math.cos(self.link1.rotation) + self.link2.length * math.cos(self.link1.rotation + self.link2.rotation), self.link1.pos[1][1] + self.link1.length * math.sin(self.link1.rotation) + self.link2.length * math.sin(self.link1.rotation + self.link2.rotation)) \
                        ]

    def inverse_kinematics(self, desired_x, desired_y):
        #c2 needs to be in [-1,1], otherwise point is outside reachable workspace
        c2 = (desired_x**2 + desired_y**2 - self.link1.length**2 - self.link2.length**2) / (2*self.link1.length * self.link2.length)
        if c2 < -1 or c2 > 1:
            return None, None

        s2 = -math.sqrt(1 - c2**2)


        theta2 = math.atan2(s2, c2)
        theta1 = math.atan2(desired_y, desired_x) - math.atan2(self.link2.length * s2, self.link1.length + self.link2.length * c2)

        return theta1, theta2



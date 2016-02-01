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

    def createBodies(self, world, tool_parameters=[ (100.0, 0.0), (50.0, math.pi / 2.0) ]):
        self.link1.pos = [ (self.basex, self.basey) ]
        self.link2.pos = [ (self.basex+self.link1.length-20, self.basey) ]


        self.link1.createBody(world, "link1", density=2.0)
        self.link2.createBody(world, "link2", density=0.5)

        self.set_pivot_positions()

        self.pivot1 = world.CreateKinematicBody(position=self.pivot_position1)
        circle=b2CircleShape(pos=self.pivot1.position, radius=1.0)
        self.pivot1.CreateFixture(shape=circle, density=0.0, friction=0.0)
        self.pivot1.userData = "pivot1"

        self.joint1 = world.CreateRevoluteJoint(bodyA=self.pivot1, bodyB=self.link1.body, anchor=self.pivot1.position, enableMotor=True, maxMotorTorque=1000000000, motorSpeed=0.0)
        self.joint2 = world.CreateRevoluteJoint(bodyA=self.link1.body, bodyB=self.link2.body, anchor=self.pivot_position2, enableMotor=True, maxMotorTorque=100000000, motorSpeed=0.0, enableLimit=True, lowerAngle=-math.pi/1.2, upperAngle=math.pi/1.2)


        self.tool = Tool(self.pivot_position3[0], self.pivot_position3[1], world, tool_parameters[0][0], body_params=tool_parameters)
        self.joint3 = world.CreateRevoluteJoint(bodyA=self.link2.body, bodyB=self.tool.bodies[0], anchor=self.pivot_position3, enableMotor=True, maxMotorTorque=100000000, motorSpeed=0.0, enableLimit=True, lowerAngle=-math.pi/1.5, upperAngle=math.pi/1.5)


    def set_pivot_positions(self):
        self.pivot_position1 = (self.basex-self.link1.length/2.0, self.basey-self.link1.line_width/4.0)

        self.pivot_position2 = (self.pivot_position1[0]+(self.link1.length * math.cos(self.link1.body.angle)), \
                                                         self.pivot_position1[1]+(self.link1.length * math.sin(self.link1.body.angle)))

        self.pivot_position3 = (self.pivot_position2[0]+((self.link2.length-2) * math.cos(self.link2.body.angle)), \
                                                         self.pivot_position2[1]+((self.link2.length-2) * math.sin(self.link2.body.angle))+5)


    def move_arm_absolute(self, theta1, theta2):
        self.link1.rotation = theta1
        self.link2.rotation = theta2

        self.link1.pos = [(self.basex, self.basey), \
                        (self.basex + self.link1.length * math.cos(self.link1.rotation), self.basey + self.link1.length * math.sin(self.link1.rotation))
                        ]
        self.link2.pos = [self.link1.pos[1], \
                        (self.link1.pos[1][0] + self.link1.length * math.cos(self.link1.rotation) + self.link2.length * math.cos(self.link1.rotation + self.link2.rotation), self.link1.pos[1][1] + self.link1.length * math.sin(self.link1.rotation) + self.link2.length * math.sin(self.link1.rotation + self.link2.rotation)) \
                        ]

    def _compute_position(self, basept, length, angle):
        p = np.empty( (2, 1) )
        p[0] = basept[0] + length * math.cos(angle)
        p[1] = basept[1] + length * math.sin(angle)

        return p

    def end_effector_position(self):
        return self._compute_position(self.pivot_position3, self.tool.length, self.joint1.angle+self.joint2.angle+self.joint3.angle)


    def inverse_kinematics_ccd(self, desired_x, desired_y):

        target = np.empty( (2, 1) )
        target[0] = desired_x
        target[1] = desired_y

        basept = np.empty( (2, 1))
        basept[0] = self.joint1.anchorA[0]
        basept[1] = self.joint1.anchorA[1]

        p1 = self._compute_position(self.joint1.anchorA, self.link1.length, self.joint1.angle)
        p2 = self._compute_position(self.joint2.anchorA, self.link2.length, self.joint1.angle+self.joint2.angle)
        p3 = self._compute_position(self.joint3.anchorA, self.tool.length, self.joint1.angle+self.joint2.angle+self.joint3.angle)

        theta1, theta2, theta3 = self.joint1.angle, self.joint2.angle, self.joint3.angle

        iteration = 0
        pi2 = math.pi * 2
        while np.linalg.norm(target - p3) > 0.02 and iteration < 20:
            dotprod = np.dot( np.transpose((p3 - p2) / np.linalg.norm(p3 - p2)), (target - p2) / np.linalg.norm(target - p2))

            delta3 = np.arccos( round(dotprod, 4) )
            delta3_direction = ((p3[0] - p2[0])*(target[1] - p2[1]) - (p3[1] - p2[1]) * (target[0] - p2[0])) \
                                / (np.linalg.norm(p3 - p2) * np.linalg.norm(target - p2))
            delta3 = delta3
            if delta3_direction < 0:
                delta3 = -delta3
            theta3 += delta3
            if theta3 > math.pi / 2.0:
                theta3 = math.pi / 2.0
            elif theta3 < -math.pi / 2.0:
                theta3 = -math.pi / 2.0
            p3 = self._compute_position(p2, self.tool.length, theta1+theta2+theta3)


            dotprod = np.dot( np.transpose((p3 - p1) / np.linalg.norm(p3 - p1)), (target - p1) / np.linalg.norm(target - p1))
            delta2 = np.arccos( round(dotprod, 4) )
            delta2_direction = ((p3[0] - p1[0])*(target[1] - p1[1]) - (p3[1] - p1[1]) * (target[0] - p1[0])) \
                                / (np.linalg.norm(p3 - p1) * np.linalg.norm(target - p1))
            delta2 = delta2
            if delta2_direction < 0:
                delta2 = -delta2
            theta2 += delta2
            if theta2 > math.pi/1.2:
                theta2 = math.pi/1.2
            elif theta2 < -math.pi/1.2:
                theta2 = -math.pi/1.2

            p2 = self._compute_position(p1, self.link2.length, theta1+theta2)
            p3 = self._compute_position(p2, self.tool.length, theta1+theta2+theta3)

            dotprod = np.dot( np.transpose((p3 - basept) / np.linalg.norm(p3 - basept)), (target - basept) / np.linalg.norm(target - basept))
            delta1 = np.arccos( round(dotprod, 4) )
            delta1_direction = ((p3[0] - basept[0])*(target[1] - basept[1]) - (p3[1] - basept[1]) * (target[0] - basept[0])) \
                                / (np.linalg.norm(p3 - basept) * np.linalg.norm(target - basept))
            delta1 = delta1
            if delta1_direction < 0:
                delta1 = -delta1
            theta1 += delta1
            theta1 = theta1 % pi2
            p1 = self._compute_position(basept, self.link1.length, theta1)
            p2 = self._compute_position(p1, self.link2.length, theta1+theta2)
            p3 = self._compute_position(p2, self.tool.length, theta1+theta2+theta3)

            iteration += 1

        return theta1, theta2, theta3
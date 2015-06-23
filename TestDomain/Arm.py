
import math
import pygame
import pygame.locals
import sys
import numpy as np


from armpart import ArmPart

class Arm:

    def __init__(self, basex, basey, length1, length2):
        self.bbox_scale = 1.0
        self.nsteps = 50
        self.link1 = ArmPart(150.0, scale=1.0, theta=(3.0/2.0) * math.pi)
        self.link2 = ArmPart(150.0, scale=1.0, theta=(1.0/2.0) * math.pi)

        self.basex = basex
        self.basey = basey

        self.move_arm_absolute(self.link1.rotation, self.link2.rotation)

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

    def move_arm_relative(self, delta1, delta2):
        self.link1.rotation += delta1
        self.link2.rotation += delta2

        self.link1.pos = [(self.basex, self.basey), \
                        (self.basex + self.link1.length * math.cos(self.link1.rotation), self.basey + self.link1.length * math.sin(self.link1.rotation))
                        ]
        self.link2.pos = [self.link1.pos[1], \
                        (self.link1.pos[1][0] + self.link1.length * math.cos(self.link1.rotation) + self.link2.length * math.cos(self.link1.rotation + self.link2.rotation), self.link1.pos[1][1] + self.link1.length * math.sin(self.link1.rotation) + self.link2.length * math.sin(self.link1.rotation + self.link2.rotation)) \
                        ]                    



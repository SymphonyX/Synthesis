import numpy as np
import pygame

import Box2D
from Box2D.b2 import *
from Box2D import *


class ArmPart:
    """
    A class for storing relevant arm segment information.
    """
    def __init__(self, length, scale=1.0):
        # some handy constants
        self.length = length
        self.scale = self.length * scale
        self.offset = self.scale / 2.0

        self.pos = None
        self.line_width = 15

    def createBody(self, world, verts, userdata="", density=0.1):
        self.body = world.CreateDynamicBody(position=(self.pos[0][0], self.pos[0][1]))
        self.vertices = verts
        shape = b2PolygonShape(vertices=self.vertices)
        self.body.CreatePolygonFixture(shape=shape, density=density, friction=1.0)
        for fix in self.body.fixtures:
            fix.sensor = True
        self.body.shape = shape
        self.body.userData = userdata







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

    def createBody(self, world, screen_height):
        self.body = world.CreateDynamicBody(position=(self.pos[0][0], screen_height-self.pos[0][1]))
        self.vertices = [(-self.length/2.0, -self.line_width/2.0), (self.length/2.0, -self.line_width/2.0), (self.length/2.0, self.line_width/2.0), (-self.length/2.0, self.line_width/2.0)]
        shape = b2PolygonShape(vertices=self.vertices)
        self.body.CreatePolygonFixture(shape=shape, density=1.0, friction=1.0)
        self.body.shape = shape







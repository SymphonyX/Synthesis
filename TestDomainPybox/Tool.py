import Box2D
from Box2D.b2 import *
from Box2D import *

class Tool:

    def __init__(self, basex, basey, world, length):
        self.basex = basex
        self.basey= basey

        self.body1 = world.CreateDynamicBody(position=(basex, basey))
        vertices = [(10.0, -5.0), (10.0, length), (-10.0, length), (-10.0, -5.0)]
        shape = b2PolygonShape(vertices=vertices)
        self.body1.CreatePolygonFixture(shape=shape, density=1.0, friction=1.0)
        self.body1.shape = shape
        self.body1.userData = "tool1"

        self.body2 = world.CreateDynamicBody(position=(self.body1.position.x, self.body1.position.y+length))
        vertices = [(10, 10.0), (-length/2.0, 10.0), (-length/2.0, -10.0), (10.0, -10.0)]
        shape = b2PolygonShape(vertices=vertices)
        self.body2.CreatePolygonFixture(shape=shape, density=0.5, friction=1.0)
        self.body2.shape = shape
        self.body2.userData = "tool2"

        self.joint1 = world.CreateRevoluteJoint(bodyA=self.body1, bodyB=self.body2, anchor=self.body1.position, enableLimit=True, lowerAngle=0.0, upperAngle=0.0)

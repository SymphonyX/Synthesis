import Box2D
from Box2D.b2 import *
from Box2D import *
import math

class Tool:

    def __init__(self, basex, basey, world, length, body_params):
        self.basex = basex
        self.basey= basey
        self.length = length

        self.bodies, self.colors, self.joints = [], [], []

        for i, body_param in enumerate(body_params):
            base_position = (basex, basey) if i == 0 else (self.bodies[-1].position.x + (body_params[i-1][0] * math.cos(body_params[i-1][1])),
                                                           self.bodies[-1].position.y + (body_params[i-1][0] * math.sin(body_params[i-1][1])))

            body = world.CreateDynamicBody(position=base_position)
            vertices = [(-5.0, -10), (body_params[i][0], -10.0), (body_params[i][0], 10.0), (-5.0, 10.0)]
            shape = b2PolygonShape(vertices=vertices)
            body.CreatePolygonFixture(shape=shape, density=0.1, friction=1.0)
            body.angle = body_params[i][1]
            body.shape = shape
            body.userData = "tool" + str(i+1)

            self.bodies.append( body )
            self.colors.append( (0, 0, 255, 0) )

            if i > 0:
                joint = world.CreateRevoluteJoint(bodyA=self.bodies[i-1], bodyB=self.bodies[i], anchor=self.bodies[i-1].position, enableLimit=True, lowerAngle=0.0, upperAngle=0.0)
                self.joints.append( joint )

        # body2_pos = ( self.body1.position.x + body_params[0][0], self.body1.position.y )
        # self.body2 = world.CreateDynamicBody(position=body2_pos)
        # vertices = [(-5.0, -10), (body_params[1][0], -10.0), (body_params[1][0], 10.0), (-5.0, 10.0)] #[(-10, -10.0), (10, -10), (10, length/2.0), (-10.0, length/2.0)]
        # shape = b2PolygonShape(vertices=vertices)
        # self.body2.angle = body_params[1][1]
        # self.body2.CreatePolygonFixture(shape=shape, density=0.01, friction=1.0)
        # self.body2.shape = shape
        # self.body2.userData = "tool2"
        #
        # self.bodies = [self.body1, self.body2]
        # self.colors = [(0, 0, 255, 0), (0, 0, 255, 0)]


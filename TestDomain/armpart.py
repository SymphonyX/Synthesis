import numpy as np
import pygame

class ArmPart:
    """
    A class for storing relevant arm segment information.
    """
    def __init__(self, length, scale=1.0, theta=0.0):
        # some handy constants
        self.length = length
        self.scale = self.length * scale
        self.offset = self.scale / 2.0

        self.rotation = theta # in radians
        self.pos = None
        line_width = 15
        arm_color = (50, 50, 50, 200) # fourth value specifies transparency

        self.surface = pygame.Surface((self.scale, line_width), pygame.SRCALPHA, 32)
        self.surface.fill(arm_color)



    def rotate(self):
        """
        Rotates and re-centers the arm segment.
        """
        # rotate our image 
        image = pygame.transform.rotozoom(self.surface, np.degrees(self.rotation), 1)
        # reset the center
        rect = image.get_rect()
        rect.center = (0, 0)

        return image, rect






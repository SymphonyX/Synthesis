import numpy as np
import pygame

class ArmPart:
    """
    A class for storing relevant arm segment information.
    """
    def __init__(self, pic, scale=1.0, theta=0.0):
        self.base = pygame.image.load(pic)
        # some handy constants
        self.length = self.base.get_rect()[2]
        self.scale = self.length * scale
        self.offset = self.scale / 2.0

        self.rotation = theta # in radians
        self.pos = None
        line_width = 15
        self.surface = pygame.Surface((self.scale, line_width), pygame.SRCALPHA, 32)


    def rotate(self, rotation):
        """
        Rotates and re-centers the arm segment.
        """
        self.rotation += rotation 
        # rotate our image 
        image = pygame.transform.rotozoom(self.base, np.degrees(self.rotation), 1)
        # reset the center
        rect = image.get_rect()
        rect.center = (0, 0)

        return image, rect






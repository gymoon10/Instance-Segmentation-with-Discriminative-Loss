import os
import numpy as np


class DataSettings(object):

    def __init__(self):

        self.CLASS_WEIGHTS = None
        # Assign it to None in order to disable class weighting

        self.MAX_N_OBJECTS = 20  # if actual # of instances < MAX_N_OBJECTS: remainders are filled with zeros

        self.N_CLASSES = 1+1

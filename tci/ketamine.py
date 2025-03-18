import numpy as np

from fus_anes.tci.tci import TCI

class TCI_Ketamine(TCI):
    def __init__(self,
                 **kw,
                 ):
        TCI.__init__(self, **kw)

    def initialize(self):
        self.k10 = 0.438
        self.k12 = 0.592
        self.k13 = 0.590
        self.k21 = 0.247
        self.k31 = 0.0146

        self.v1 = 0.063 * self.weight
        self.v2 = self.v1 * self.k12 / self.k21
        self.v3 = self.v1 * self.k13 / self.k31

        self.ke0 = 1.0 # TODO ??

        TCI.initialize_model(self)

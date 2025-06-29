##
import numpy as np
import logging

from fus_anes.util import now
import fus_anes.config as config

class TCI():
    def __init__(self,
                 age=config.age,
                 weight=config.weight,
                 height=config.height,
                 sex=config.sex,
                 blood_mode='venous',
                 opiates=False,
                 resolution=0.050,
                 prior_tcm=None):
        '''
        https://www.sciencedirect.com/science/article/pii/S0007091218300515?via%3Dihub#sec2
        '''
        self.resolution = resolution # in seconds - smallest interval TCI can handle (useless to call wait() in smaller steps
        self.resolution_mult = 1/self.resolution # 1 means 1-sec resolution, 10 means 100-msec (=1 decisecond), 100 means 10-ms, 1000 means 1ms
    
        self.age = age
        self.sex = sex
        self.weight = weight
        self.height = height
        self.blood_mode = blood_mode
        self.opiates = opiates
        self.prior_tcm = prior_tcm

        self.pre_setup()
        self.initialize()
    
    def pre_setup(self):
        
        if self.prior_tcm is not None:
            age = self.prior_tcm.pop('age')
            weight = self.prior_tcm.pop('weight')
            height = self.prior_tcm.pop('height')
            sex = self.prior_tcm.pop('sex')
            blood_mode = self.prior_tcm.pop('blood_mode')
            opiates = self.prior_tcm.pop('opiates')
            if (age != self.age) or (weight != self.weight) or (height != self.height) or (sex != self.sex) or (blood_mode != self.blood_mode) or (opiates != self.opiates):
                logging.warning(f'Aborting prior TCI load because demographics do not match ({age}, {sex}, {weight}, {height})')
                self.prior_tcm = None

        # starting values
        self.infusion_rate = 0.0
        self.x1 = 0.0 # "plasma"
        self.x2 = 0.0 # "muscle" (rapidly equilibrating)
        self.x3 = 0.0 # "fat" (slowly equilibrating)
        self.xe0 = 0.0 # "brain" (effect site)

        if self.prior_tcm is not None:
            for k,v in self.prior_tcm.items():
                setattr(self, k, v)
            prior_ts = self.prior_tcm['ts']
            dt = int(round(now(minimal=True) - prior_ts))
            for _ in range(dt):
                self.wait(1.0)
            return

    def initialize_model(self):
        time_divide = 60 * self.resolution_mult # bc these params were all in /min
        self.ke0 /= time_divide # central <-> brain
        self.k10 /= time_divide # elimination
        self.k12 /= time_divide # central <-> rapidly equilibrating
        self.k13 /= time_divide # central <-> slowly equilibrating
        self.k21 /= time_divide # rapidly equilibrating <-> central
        self.k31 /= time_divide # slowly equilibrating <-> central


    @property
    def level(self):
        return self.xe0
    @property
    def cp(self):
        return self.x1
    @property
    def ce(self):
        return self.xe0

    def bolus(self, dose):
        # dose in mg
        self.x1 += dose / self.v1
    
    def infuse(self, dose):
        # dose in mcg/kg.min
        self.infusion_rate = dose

    def wait(self, secs):
        n_steps = int(np.round(self.resolution_mult * secs)) # in correct unit

        dose_per_sec = self.infusion_rate * self.weight / (1000*60) # mcg/kg/min to mg/sec
        dose_per_step = dose_per_sec / self.resolution_mult

        for _ in range(n_steps):
            if dose_per_step > 0:
                self.bolus(dose_per_step)

            x1k10 = self.x1 * self.k10
            x1k12 = self.x1 * self.k12
            x1k13 = self.x1 * self.k13
            x2k21 = self.x2 * self.k21
            x3k31 = self.x3 * self.k31

            xk1e = self.x1 * self.ke0
            xke1 = self.xe0 * self.ke0

            self.x1 += x2k21 - x1k12 + x3k31 - x1k13 - x1k10
            self.x2 += x1k12 - x2k21
            self.x3 += x1k13 - x3k31
            self.xe0 += xk1e - xke1
    
    def export(self):
        to_export = ['x1', 'x2', 'x3', 'xe0',
                     'v1', 'v2', 'v3',
                     'k10', 'k12', 'k13', 'k21', 'k31', 'ke0']
        to_export = {k: getattr(self, k) for k in to_export}

        to_export.update(age=self.age,
                 sex=self.sex,
                 height=self.height,
                 weight=self.weight,
                 opiates=self.opiates,
                 blood_mode=self.blood_mode)
        return to_export

##

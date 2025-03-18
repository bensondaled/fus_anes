import numpy as np

from fus_anes.tci.tci import TCI

class TCI_Propofol(TCI):
    def __init__(self,
                 opiates=False,
                 blood_mode='venous',
                 **kw,
                 ):

        self.opiates = opiates
        self.blood_mode = blood_mode

        TCI.__init__(self, **kw)
        

    def initialize(self):
        age = self.age
        sex = self.sex
        weight = self.weight
        height = self.height
        opiates = self.opiates
        blood_mode = self.blood_mode

        age_ref = 35
        weight_ref = 70.0
        height_ref = 170
        sex_ref = 'm'

        ibw = 22 * height**2
        dose_weight = ibw + 0.4 * (weight - ibw)
        bmi = weight / (height / 100)**2
        bmi_ref = weight_ref / (height_ref / 100)**2
        post_menstrual_age = age*52 + 40 #"If PMA was not recorded, it was assumed to be 40 weeks longer than age." - note age is in years and pma is in weeks, per paper
        post_menstrual_age_ref = age_ref*52 + 40

        theta_1 = 6.28 # v1_ref
        theta_2 = 25.5 # v2_ref
        theta_3 = 273 # v3_ref
        theta_4 = 1.79 # cl_ref (male)
        theta_5 = 1.75 # q2_ref
        theta_6 = 1.11 # q3_ref
        theta_7 = 0.191 # typical residual error
        theta_8 = 42.3 # cl maturation e50
        theta_9 = 9.06 # cl maturation slope
        theta_10 = -0.0156 # smaller V2 with age
        theta_11 = -0.00286 # lower CL with age
        theta_12 = 33.6 # weight for 50% of maximal V1
        theta_13 = -0.0138 # smaller V3 with age
        theta_14 = 68.3 # maturation of Q3
        theta_15 = 2.10 # cl_ref (female)
        theta_16 = 1.30 # higher q2 for maturation of q3
        theta_17 = 1.42 # v1 venous samples (children)
        theta_18 = 0.68 # higher q2 venous samples

        f_ageing = lambda x, age: np.exp(x * (age - age_ref))
        f_sigmoid = lambda x, e50, lamb: (x ** lamb) / (x ** lamb + e50 ** lamb)
        f_central = lambda x: f_sigmoid(x, theta_12, 1)
        f_cl_maturation = lambda pma: f_sigmoid(pma, theta_8, theta_9)
        f_q3_maturation = lambda age: f_sigmoid((age*52+40), theta_14, 1) # unclear in paper if age here is years or weeks, but looks like weeks based on table 2
        f_opiates = lambda x: 1 if opiates==False else np.exp(x * age)
        def f_al_sallami(age, weight, bmi, sex):
            if sex == 'm':
                return (0.88 + (1-0.88) / (1 + (age / 13.4)**-12.7)) * ((9270 * weight) / (6680 + 216 * bmi))
            elif sex == 'f':
                return (1.11 + (1-1.11) / (1 + (age / 7.1)**-1.1)) * ((9270 * weight) / (8780 + 244 * bmi))

        v1_arterial = theta_1 * (f_central(weight) / f_central(weight_ref)) #* np.exp(eta_1) # liters
        v1_venous = v1_arterial * (1 + theta_17 * (1 - f_central(weight)))
        v2 = theta_2 * (weight / weight_ref) * f_ageing(theta_10, age) #* np.exp(eta_2)
        v2_ref = theta_2 * (weight_ref / weight_ref) * f_ageing(theta_10, age_ref) #* np.exp(eta_2)
        v3 = theta_3 * (f_al_sallami(age, weight, bmi, sex) / f_al_sallami(age_ref, weight_ref, bmi_ref, sex_ref)) * f_opiates(theta_13) #* np.exp(eta_3)
        v3_ref = theta_3 * (f_al_sallami(age_ref, weight_ref, bmi_ref, sex_ref) / f_al_sallami(age_ref, weight_ref, bmi_ref, sex_ref)) * f_opiates(theta_13) #* np.exp(eta_3)
        _cl_theta = theta_4 if sex=='m' else theta_15
        cl = _cl_theta * ((weight / weight_ref)**0.75) * (f_cl_maturation(post_menstrual_age) / f_cl_maturation(post_menstrual_age_ref)) * f_opiates(theta_11) #* np.exp(eta_4)
        q2_arterial = theta_5 * ((v2 / v2_ref)**0.75) * (1 + theta_16 * (1 - f_q3_maturation(age))) 
        q2_venous = q2_arterial * theta_18
        q3 = theta_6 * ((v3 / v3_ref)**0.75) * (f_q3_maturation(age) / f_q3_maturation(age_ref)) #* np.exp(eta_6)
        
        if self.blood_mode == 'arterial':
            ke0 = 0.146 * ((weight / 70.0) ** -0.25) # theta_2 from table 3
        elif self.blood_mode == 'venous':
            ke0 = 1.24 * ((weight / 70.0) ** -0.25) # theta_8 from table 3

        # volumes
        self.v1 = v1_arterial if self.blood_mode=='arterial' else v1_venous
        self.v2 = v2
        self.v3 = v3

        # rate constants
        self.ke0 = ke0
        self.k10 = cl / self.v1
        self.k12 = q2_arterial / self.v1 if self.blood_mode=='arterial' else q2_venous / self.v1
        self.k13 = q3 / self.v1
        self.k21 = (self.k12 * self.v1) / self.v2
        self.k31 = (self.k13 * self.v1) / self.v3

        TCI.initialize_model(self)

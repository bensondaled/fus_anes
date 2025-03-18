import os
import json
import pandas as pd
import numpy as np

import fus_anes.config as config
from .tci_util import bolus_to_infusion, hold_at_level, compute_bolus_to_reach_ce

class TCIProtocol():
    '''Generates a series of instructions for a TCI pump to achieve specified effect site propofol concentrations

    A given protocol is defined as an initial weight-based bolus followed by adjustments to an infusionin timesteps specified by the resolution parameter.

    model_params : supplied to Schnider propofol TCI model
    initial_bolus : in mg/kg, given once as first instruction
    bolus_refractory_period : time in seconds after bolus during which no infusion can be started
    target_values : list of target effect site concentrations of propofol (Ce)
    target_durations : parallel to target_values, list of durations (non-cumulative time) at which to hold effect site concentration at the specified value (e.g. target_durations[i]==180 will aim to hold Ce at target_values[i] for 3 minutes, starting right after the previous entry)
    resolution : time interval in secs at which new instructions are given to pump (lower leads to higher-resolution adjustments)

    '''
    BOLUS, INFUSION = 0, 1
    def __init__(self, 
                 preexisting_tci_obj=None,
                 model_params=dict(age=config.age,
                                   weight=config.weight,
                                   height=config.height,
                                   sex=config.sex),
                 initial_bolus=None, # mg/kg
                 initial_bolus_ce=None,
                 bolus_refractory_period=60*2, # secs; even if 0, the time to deliver bolus is effectively a refractory period where no infusion instructions will be generated
                 target_values=None,
                 target_durations=None,
                 resolution=15, # secs,
                 foresight=4, # see hold_at_level
                 ):
        
        self.params = dict(
                 preexisting_tci_obj=preexisting_tci_obj,
                 model_params=model_params,
                 initial_bolus=initial_bolus,
                 initial_bolus_ce=initial_bolus_ce,
                 bolus_refractory_period=int(round(bolus_refractory_period)),
                 target_values=target_values,
                 target_durations=target_durations,
                 resolution=resolution,
                 foresight=foresight,
                 )
        # store params into class vars
        for name, val in self.params.items():
            setattr(self, name, val)
        self.instructions = None
        self._idx = 0
    

    def step(self):
        if self.instructions is None:
            return None
        
        self._idx += 1

    @property
    def current_step(self):
        if self._idx >= len(self.instructions):
            return None
        return self.instructions.iloc[self._idx]
        
    def generate(self):
        assert isinstance(self.target_values, (list, np.ndarray))
        assert isinstance(self.target_durations, (list, np.ndarray))
        assert len(self.target_values) == len(self.target_durations)

        def accurate_wait(tci_obj, dur):
            # dur in secs
            vals = []
            nsteps = int(round(dur * 10))
            for _ in range(nsteps):
                tci_obj.wait(0.1)
                vals.append(tci_obj.level)
            return vals

        self.simulated_ce = [] # simulated effect site concentration in steps of 1 sec
        self.instructions = []
        dt = 0 # elapsed time counter

        # initialize TCI model
        if self.preexisting_tci_obj is None:
            self.tci = TCI(**self.model_params)
        else:
            self.tci = self.preexisting_tci_obj
        self.simulated_ce.append(self.tci.level)
        
        # deliver initial bolus 
        if self.initial_bolus is not None:
            initial_bolus_mg = self.initial_bolus * self.model_params['weight']
        if self.initial_bolus_ce is not None: # NOTE this takes priority if both are non-None
            initial_bolus_mg = compute_bolus_to_reach_ce(self.tci, self.initial_bolus_ce, initial_guess=1.0*self.model_params['weight'])
        bolus_as_inf_dur, bolus_as_inf_rate = bolus_to_infusion(initial_bolus_mg)

        self.tci.infuse(bolus_as_inf_rate)
        self.simulated_ce += accurate_wait(self.tci, bolus_as_inf_dur)
        self.tci.infuse(0)
        
        extra_wait = self.bolus_refractory_period - bolus_as_inf_dur
        if extra_wait > 0:
            self.simulated_ce += accurate_wait(self.tci, extra_wait)
            dt += extra_wait
        
        self.instructions.append(dict(start_time=0,
                                      kind=self.BOLUS,
                                      dose=initial_bolus_mg,
                                      target_idx=0,
                                      target_val=None,
                                      target_dur=dt,
                                    ))

        # iterate through targets, generating instruction set for each one
        for tidx, (ttime, tval) in enumerate(zip(self.target_durations, self.target_values)):
            print(f'Computing target {tidx+1} / {len(self.target_durations)}')
            print(f'\tGoal {tval} for {ttime}s, current rate={self.tci.infusion_rate}')

            rates = hold_at_level(self.tci, tval, ttime,
                                  resolution=self.resolution,
                                  foresight=self.foresight,
                                  )

            for rate in rates:
                self.tci.infuse(rate)
                self.instructions.append(dict(start_time=dt,
                                              kind=self.INFUSION,
                                              dose=rate,
                                              target_idx=tidx+1,
                                              target_val=tval,
                                              target_dur=self.resolution,
                                            ))
                self.simulated_ce += accurate_wait(self.tci, self.resolution)
                dt += self.resolution

        self.simulated_ce = pd.Series(self.simulated_ce)
        self.instructions = pd.DataFrame(self.instructions)
        self.name = 'unnamed'
        
    def save(self, save_path):
    # save protocol with all relevant info
        param_txt = json.dumps({k:v for k,v in self.params.items() if 'tci' not in k})
        if not save_path.endswith('.h5'):
            save_path = f'{save_path}.h5'
            
        ei = 0
        spname = os.path.splitext(save_path)[0]
        while os.path.exists(save_path):
            save_path = f'{spname}_{ei}.h5'
            ei += 1
            
        with pd.HDFStore(save_path) as h:
            h.put('params', pd.Series(param_txt))
            h.put('simulated_ce', self.simulated_ce)
            h.put('instructions', self.instructions)


    def load(self, load_path):

        if load_path.endswith('.h5'):
            # h5 files are already-generated protocols
            if self.target_durations is not None or self.target_values is not None:
                logging.warning('Supplied targets overwritten by load.')

            with pd.HDFStore(load_path) as h:
                self.params = h.params
                self.simulated_ce = h.simulated_ce
                self.instructions = h.instructions
            self.params = json.loads(self.params.values[0])
            for name, val in self.params.items():
                setattr(self, name, val)
            self.name = os.path.split(load_path)[-1]

        elif load_path.endswith('.py'):
            # py files are to be parsed and used to generate a new protocol: they should be proper python dicts with each desired init param specified, rest will default to class defaults
            # but cannot do things like imports in those files, bc will be "eval"ing them not importing them (reason for that is ability to dynamically edit while running)
            with open(load_path, 'r') as f:
                info = f.read()
            try:
                info = eval(info)
            except:
                print(f'Issue loading {load_path}, aborting protocol generation.')
                return None
            
            for pname, param in info.items():
                self.params[pname] = param
            for name, val in self.params.items():
                setattr(self, name, val)

            self.generate()
            self.name = os.path.split(load_path)[-1]
            
            # save copy
            copy_path = os.path.split(load_path)[0]
            nm = os.path.splitext(self.name)[0]
            copy_path = os.path.join(copy_path, f'{nm}_generated.h5')
            self.save(copy_path)

        return self


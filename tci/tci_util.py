import copy
import numpy as np
from scipy.optimize import minimize

import fus_anes.config as config

##
def simulate(tci_obj, bolus=None, infusion=0, dur=10, sim_resolution=0.200, report_resolution=None, report_fxn=lambda x: x[-1], return_sim_object=False):
    '''Sim resolution being small is somewhat important because boluses run so fast that every few milliseconds count

    Report resolution: even if the sim is run at high resolution like 1ms per step, you can choose to aggregate values to any reported resolution.
        if None: uses sim_resolution
        if 'dur': uses the supplied dur (so gives one output for each dur given, corresponding to report_fxn of that duration)

    Report fxn: consider np.mean, np.max/min, or a custom lambda x: x[-1], to use last value
    '''
    NUM = (int, float, np.int64, np.int32, np.int16, np.int8, np.float16, np.float32, np.float64)

    if isinstance(dur, NUM):
        dur = [dur]

        if infusion is None:
            infusion = 0 # maybe consider nan and handling that later so it can keep its rate?
        assert isinstance(infusion, NUM)
        infusion = [infusion]

        if bolus is None:
            bolus = (0, 0)
        assert isinstance(bolus[0], NUM)
        bolus = [bolus]
    else:
        if infusion is None:
            infusion = [0] * len(dur)
        if bolus is None:
            bolus = [(0,0)] * len(dur)

    assert isinstance(dur, (list, tuple, np.ndarray))
    assert isinstance(infusion, (list, tuple, np.ndarray))
    assert isinstance(bolus[0], (list, tuple, np.ndarray))

    obj = copy.deepcopy(tci_obj)

    result = []
    elapsed = 0.0

    for _bolus, _infusion, _dur in zip(bolus, infusion, dur):
        step_results = []

        _dur = float(_dur)
        _infusion = float(_infusion)
        
        ''' plan to remove boluses enrtirely
        # bolus
        if not (_bolus[0]==0 and _bolus[1]==0):
            bdur, bdose = _bolus
            bdur = round(bdur, 2) # 10millisecond resolution hard-coded for bolus duration accuracy
            obj.infuse(bdose)
            for _ in np.arange(0, bdur, sim_resolution):
                step_results.append(obj.level)
                obj.wait(sim_resolution)
                elapsed += sim_resolution
        '''

        # infusion
        obj.infuse(_infusion)
        remaining = _dur - elapsed
        for _ in np.arange(0, remaining, sim_resolution):
            obj.wait(sim_resolution)
            step_results.append(obj.level)

        if report_resolution is None:
            result += step_results
        elif report_resolution == 'dur':
            result.append(report_fxn(step_results))
    
    if return_sim_object:
        return np.array(result), obj
    return np.array(result)
##
def bolus_to_infusion(dose):
    # dose in mg
    # convenience for calculating how a bolus is practically achieved as a temporary infusion
    ml_required = dose / 10 # bc propofol 10mg/ml
    mins_required = ml_required / config.bolus_rate
    secs_required = mins_required * 60.0

    mg_min = config.bolus_rate * 10 # bc propofol 10mg/ml
    mcg_min = mg_min * 1000
    mcg_kg_min = mcg_min / config.weight

    return secs_required, mcg_kg_min

def compute_bolus_to_reach_ce(tci_obj, target_ce, initial_guess=10, lookahead=60*5,
                              search_range=[0, 250], max_iters=10000, lamb=5, error_tolerance=0.1):
    b_dose = initial_guess
    error = 1e3
    niters = 0

    while abs(error) > error_tolerance:
        secs, i_dose = bolus_to_infusion(b_dose)
        outcome = np.max(simulate(tci_obj, (secs, i_dose), 0, lookahead))
        
        error = outcome - target_ce
        b_dose = b_dose - lamb * error
        #print(b_dose, error)

        if b_dose >= search_range[1]:
            b_dose = search_range[1]
            break
        if b_dose <= search_range[0]:
            b_dose = search_range[0]
            break

        niters += 1
        if niters > max_iters:
            break
    #print(niters)
    return b_dose

##
def compute_bolus_to_reach_ce_2(tci_obj, target, max_dur=60*5):
    
    bolus_mg_min = config.bolus_rate * 10 # bc propofol 10mg/ml
    bolus_mcg_min = bolus_mg_min * 1000
    bolus_mcg_kg_min = bolus_mcg_min / config.weight
    bolus_rate = bolus_mcg_kg_min

    def err_fxn(bolus_duration, tci_obj, target_val):
        bolus_duration = bolus_duration[0] # for minimize mechanics

        durs = [bolus_duration, max_dur-bolus_duration]
        rates = [bolus_rate, tci_obj.infusion_rate]

        traj = simulate(tci_obj,
                      infusion=rates,
                      dur=durs,
                      bolus=None,
                      sim_resolution=0.050,
                      report_resolution=None)
        outcome = np.max(traj)
        err = np.abs(outcome - target_val)
        return err

    initial_guess = np.array([5.0]) # giving secs of bolus time
    opt = minimize(err_fxn,
                   initial_guess,
                   method='L-BFGS-B',
                   args=(tci_obj, target),
                   options=dict(eps=0.050, ftol=1e-20, gtol=1e-20),
                   bounds=[(0.100, 60)]
                   )

    result = opt.x[0]
    return result, bolus_rate


##

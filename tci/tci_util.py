import copy
import numpy as np

import fus_anes.config as config

def simulate(tci_obj, bolus=(0, 0), infusion=0, dur=10):
    obj = copy.deepcopy(tci_obj)
    dur = int(round(dur))
    result = []
    
    # bolus
    if not (bolus[0]==0 and bolus[1]==0):
        bdur, bdose = bolus
        bdur = int(round(bdur))
        obj.infuse(bdose)
        for i in range(bdur):
            result.append(obj.level)
            obj.wait(1)

    # infusion
    obj.infuse(infusion)
    for i in range(max(0, dur - len(result))):
        result.append(obj.level)
        obj.wait(1)

    return np.array(result)

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

def compute_infusion_rate_for_target(tci_obj, target, duration,
                                     search_range=[0, 300],
                                     initial_guess=None,
                                     lamb=10,
                                     error_tolerance=0.001,
                                     max_iters=10000):
    
    # using the current state of tci_obj, return optimal rate to achieve target level by duration time from now
    # if initial guess is None, use current infusion rate of tci_obj
    error = 1e3
    niters = 0
    rate = initial_guess
    if rate is None:
        rate = tci_obj.infusion_rate
    while abs(error) > error_tolerance:
        outcome = simulate(tci_obj, (0,0), rate, duration)[-1]

        error = outcome - target
        rate = rate - lamb * error
        #print(error, rate)

        if rate >= search_range[1]:
            rate = search_range[1]
            break
        if rate <= search_range[0]:
            rate = search_range[0]
            break

        niters += 1
        if niters > max_iters:
            break
    #print(niters)
    return rate

def hold_at_level(tci_obj, target, duration,
                  resolution=30,
                  foresight=1,
                  **computation_kw):
    # return list of infusion rates to hold tci_obj at target concentration 
    # list will be at resolution in seconds for specified duration
    # foresight: look n resolution-size steps into future, calculating optimal rate at each one, and take median of results. intuitively, this tradeoff is: lower will be a "short-term thinker" where you arrive fast at your target but are likely to overshoot; higher will be a "long-term thinker" where you are more steady in long-run but sacrifice the speed at which you get there
    rates = []
    computation_kw = computation_kw.copy()
    computation_kw['initial_guess'] = computation_kw.pop('initial_guess', tci_obj.infusion_rate)

    for step in np.arange(resolution, duration+resolution, resolution):
        #print(computation_kw['initial_guess'])
        test_dur = [resolution*i for i in range(1, foresight+1)]
        rate = [compute_infusion_rate_for_target(tci_obj, target, duration=r, **computation_kw) for r in test_dur]
        #print(computation_kw['initial_guess'])
        #print(rate)
        rate = np.median(rate)
        rates.append(rate)
        tci_obj = copy.deepcopy(tci_obj)
        tci_obj.infuse(rate)
        tci_obj.wait(resolution)
        computation_kw['initial_guess'] = rate # for efficiency, helps tremendously
    return rates

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


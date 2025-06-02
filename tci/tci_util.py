import copy
import numpy as np

import fus_anes.config as config

def simulation_error(rate, target, end_time, tci_obj):
    reached = simulate()
    error = reached - target
    return error


def simulate(tci_obj, bolus=(0, 0), infusion=0, dur=10, sim_resolution=0.200):
    '''Sim resolution being small is somewhat important because boluses run so fast that every few milliseconds count
    '''

    if isinstance(dur, (int, float)):
        dur = [dur]

        if infusion is None:
            infusion = 0 # maybe consider nan and handling that later so it can keep its rate?
        assert isinstance(infusion, (int, float))
        infusion = [infusion]

        if bolus is None:
            bolus = (0, 0)
        assert isinstance(bolus[0], (int, float))
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
    elapsed = 0

    for _bolus, _infusion, _dur in zip(bolus, infusion, dur):
        _dur = int(round(_dur))
        
        # bolus
        if not (_bolus[0]==0 and _bolus[1]==0):
            bdur, bdose = _bolus
            bdur = round(bdur, 1) # 100millisecond resolution hard-coded for bolus duration accuracy
            obj.infuse(bdose)
            for i in np.arange(0, bdur+sim_resolution, sim_resolution):
                result.append(obj.level)
                obj.wait(sim_resolution)
                elapsed += sim_resolution

        # infusion
        obj.infuse(_infusion)
        remaining = _dur - elapsed
        for i in np.arange(0, remaining+sim_resolution, sim_resolution):
            result.append(obj.level)
            obj.wait(sim_resolution)
    
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


import copy
import numpy as np
from scipy.optimize import minimize

import fus_anes.config as config

def mlmin_mcgkgmin(mlmin=None, mcgkgmin=None):
    if mlmin is not None:
        mg_min = mlmin * config.drug_mg_ml
        mcg_min = mg_min * 1000
        mcg_kg_min = mcg_min / config.weight
        return mcg_kg_min
    elif mcgkgmin is not None:
        mcg_min = mcgkgmin * config.weight
        mg_min = mcg_min / 1000
        mlmin = mg_min / config.drug_mg_ml
        return mlmin

def bolus_to_infusion(dose):
    # dose in mg
    # convenience for calculating how a bolus is practically achieved as a temporary infusion
    ml_required = dose / config.drug_mg_ml # bc propofol 10mg/ml
    mins_required = ml_required / config.bolus_rate
    secs_required = mins_required * 60.0
    
    mcg_kg_min = mlmin_mcgkgmin(mlmin=config.bolus_rate)

    return secs_required, mcg_kg_min

##
def simulate(tci_obj,
             infusion=[0],
             dur=[10],
             sim_resolution=None,
             report_resolution=None,
             report_fxn=lambda x: x[-1],
             return_sim_object=False):
    '''Sim resolution being small is somewhat important because boluses run so fast that every few milliseconds count

    Report resolution: even if the sim is run at high resolution like 1ms per step, you can choose to aggregate values to any reported resolution.
        if None: uses sim_resolution
        if 'dur': uses the supplied dur (so gives one output for each dur given, corresponding to report_fxn of that duration)

    Report fxn: consider np.mean, np.max/min, or a custom lambda x: x[-1], to use last value
    '''
    if sim_resolution is None:
        sim_resolution = tci_obj.resolution

    if sim_resolution < tci_obj.resolution:
        print('Adjusting sim resolution to be at least TCI resolution')
        sim_resolution = tci_obj.resolution

    NUM = (int, float, np.int64, np.int32, np.int16, np.int8, np.float16, np.float32, np.float64)

    if isinstance(dur, NUM):
        dur = [dur]

        if infusion is None:
            infusion = 0 # maybe consider nan and handling that later so it can keep its rate?
        assert isinstance(infusion, NUM)
        infusion = [infusion]

    else:
        if infusion is None:
            infusion = [0] * len(dur)

    assert isinstance(dur, (list, tuple, np.ndarray))
    assert isinstance(infusion, (list, tuple, np.ndarray))

    obj = copy.deepcopy(tci_obj)

    result = []
    elapsed = 0.0

    for _infusion, _dur in zip(infusion, dur):
        step_results = []

        _dur = float(_dur)
        _infusion = float(_infusion)
        
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

def compute_bolus_to_reach_ce(tci_obj, target, max_dur=60*5, sim_resolution=None):

    bolus_mcg_kg_min = mlmin_mcgkgmin(mlmin=config.bolus_rate)

    if sim_resolution is None:
        sim_resolution = tci_obj.resolution

    def err_fxn(bolus_duration, tci_obj, target_val):
        bolus_duration = bolus_duration[0] # for minimize mechanics

        durs = [bolus_duration, max_dur-bolus_duration]
        rates = [bolus_mcg_kg_min, tci_obj.infusion_rate]

        traj = simulate(tci_obj,
                        infusion=rates,
                        dur=durs,
                        sim_resolution=sim_resolution,
                      )
        outcome = np.max(traj)
        err = np.abs(outcome - target_val)
        return err

    initial_guess = np.array([5.0]) # giving secs of bolus time
    opt = minimize(err_fxn,
                   initial_guess,
                   method='L-BFGS-B',
                   args=(tci_obj, target),
                   options=dict(eps=sim_resolution*2, ftol=1e-20, gtol=1e-20), # need to make eps greater than sim_resolution bc otherwise it tries tiny little steps that make by definition no difference to the ismulation, and no change occurs so it assumes optimized
                   bounds=[(0.100, 60.0)]
                   )

    secs_bolus = opt.x[0]
    return secs_bolus, bolus_mcg_kg_min


def compute_optimal_rates(target_levels,
                          target_durations,
                          tci_obj,
                          instruction_interval=15.0,
                          sim_resolution=1.0,
                          ):
    '''This function has the strict goal of simultaneously optimizing a series of consecutive infusion rates, in order to minimize the error of reaching each target supplied at each duration supplied.

    Target levels: ce goals
    Target durations: secs from LAST target
    Instruction interval: secs between rate changes (does NOT necessarily need to correspond to target durations in any way - rather this determines the spacing of uniform instructions that will be made)
    Sim resolution: time resolution of internal simulation used to compute results, in secs
    '''
    max_rate = mlmin_mcgkgmin(mlmin=config.max_rate) # in mcg/kg/min
    
    # this is a special step that adds 1 extra interval on so it doesnt get lazy with final instruction
    # it inherently assumes you'll want to stay where you ended
    target_durations = np.append(target_durations, instruction_interval)
    target_levels = np.append(target_levels, target_levels[-1])
    
    # expand out the range of time over which we'll be simulating into uniform chunks that will be able to have instruction changes
    end_target_time = np.sum(target_durations)
    ttimes = np.arange(0, end_target_time+1, instruction_interval)
    if ttimes[-1] != end_target_time:
        ttimes = np.append(ttimes, end_target_time)
    
    durations = np.diff(ttimes)
    
    # prepare which of the expanded simulated values actually get evaluated for accuracy
    evaluation_indices = [np.argmin(np.abs(np.cumsum(durations) - td)) for td in np.cumsum(target_durations)]
    evaluation_indices = [ei for i,ei in enumerate(evaluation_indices) if target_levels[i] is not None]
    evaluation_values = [ev for ev in target_levels if ev is not None]

    def simulation_error(rates, eval_targets, eval_idxs, durations, tci_obj):
        traj = simulate(tci_obj,
                        infusion=rates,
                        dur=durations,
                        sim_resolution=sim_resolution,
                        report_resolution='dur',
                        report_fxn=lambda x: x[-1])

        traj_eval = traj[eval_idxs]
        error = np.abs(traj_eval - eval_targets)
        error = np.sum(error)

        return error
    
    initial_guess_rates = np.array([tci_obj.infusion_rate] * len(durations))
    opt = minimize(simulation_error,
                   initial_guess_rates,
                   method='L-BFGS-B',
                   args=(evaluation_values, evaluation_indices, durations, tci_obj),
                   options=dict(ftol=1e-5),
                   bounds=[(0, max_rate)])

    rates = opt.x
    durs = durations
    
    # now cut off the special "laziness preventing" extra interval
    rates = rates[:-1]
    durs = durs[:-1]

    return rates, durs

def go_to_target(tci_obj, target,
                 delta_thresh=0.05,
                 new_target_travel_time=1*60.0,
                 duration=60*6,
                 step_resolution=15.0):
    '''This function uses compute_optimal_rates internally in order to create a plan for reaching and maintaining a target. It determines how as follows:

    - If the new target appears to be a change, we set a goal to jump there
    - If the new target appears to be stable, we just compute more timesteps for it

    In all cases, we return instructions for the total duration we want all these instructions to go for (once reaching target, we hold stable until duration is done)

    target: ce level
    delta_thresh: what ce change from current is considered different enough to be a "new" level
    new_target_travel_time: seconds dedicated to reaching new target before staying
    duration: total seconds of all instructions returned combined
    step_resolution: of instructions returned

    It's in some sense a wrapper to compute_optimal_rates bc that's what does all the work, but using that function directly can take very long if you don't specify reasonable instruction sets (eg if you try to optimize for a 15-minute period all in one big step; because that function is a pure optimizer with no knowledge of practical goals. So this breaks it into feasible and realistic steps.
    '''


    current_level = tci_obj.level
    delta = target - current_level

    time_used = 0
    all_rates = []
    all_durs = []

    # Step 1: only if new level is a jump: we aim to get there in the specified time, then hold for a minute, which helps ensure stability when we arrive.
    if np.abs(delta) > 0.05:

        assert new_target_travel_time + 1 <= duration, 'Didnt allow enough time to travel to destination, need travel_time+1 total duration at minimum, in order to get there and hold stable'

        # in this approach for a new target, we give travel_time to get there, then ensure a minute of stability as part of the optimization
        n_res_to_1min = int(60.0 // step_resolution)
        target_durations = [new_target_travel_time] + [step_resolution]*n_res_to_1min
        target_levels = [target] * len(target_durations)
        rates, durs = compute_optimal_rates(target_levels,
                                            target_durations,
                                            tci_obj,
                                            instruction_interval=step_resolution,
                                            )

        time_used += np.sum(durs)
        all_rates.append(rates)
        all_durs.append(durs)

        # implement these steps in simulation
        _, tci_obj = simulate(tci_obj,
                              infusion=rates,
                              dur=durs,
                              return_sim_object=True)
    
    # Step 2: the stability step (sometimes this is all this function is for) - it just breaks the task of holding steady into small step_resolution-sized chunks of holding steady (and note we didnt have to match the time resolution of these steps with the time resolution for compute_optimal_rates, but we do for convenience). The point is that if you just say "hold steady at this level for 10 mins" directly to compute_optimal_rates, it could give you tons of oscillations in the middle between time 0 and 10. This wrapping of it in small steps is (a) more efficienct, and (b) ensures that every few seconds we're still optimzing to be steady at that level.
    time_remaining = duration - time_used
    stability_nsteps = int(time_remaining // step_resolution)
    target_durations = [step_resolution] * stability_nsteps

    target_levels = [target] * len(target_durations)
    rates, durs = compute_optimal_rates(target_levels,
                                        target_durations,
                                        tci_obj,
                                        instruction_interval=step_resolution
                                        )
    all_rates.append(rates)
    all_durs.append(durs)

    rates = np.concatenate(all_rates)
    durs = np.concatenate(all_durs)
    return rates, durs
##

##
import fus_anes.config as config
from fus_anes.tci import TCI_Propofol as TCI
from fus_anes.tci.tci_util import compute_bolus_to_reach_ce, bolus_to_infusion, simulate, compute_bolus_to_reach_ce_2
from scipy.optimize import minimize
import copy

age = config.age
weight = config.weight # kg
height = config.height # cm
sex = config.sex

def collect_vals(t, dur=10*60):
    # dur in secs
    vals = []
    nsteps = int(round(dur))
    for _ in range(nsteps):
        t.wait(1.0)
        vals.append((t.cp, t.ce))
    return vals


# Test #1: comparing to shafer with specific arbitrary scenarios
'''
t = TCI(age=age, weight=weight, height=height, sex=sex, blood_mode='venous', opiates=True)

vals = collect_vals(t, dur=10*60)
t.bolus(50)
t.infuse(125)
vals += collect_vals(t, dur=40*60)
t.infuse(75)
vals += collect_vals(t, dur=30*60)
t.bolus(20)
t.infuse(0)
vals += collect_vals(t, dur=60*60)

cp, ce = np.array(vals).T

pl.figure()
pl.plot(np.arange(len(cp))/60, cp, label='cp', color='gold', ls='--')
pl.plot(np.arange(len(ce))/60, ce, label='ce', color='gold',)
pl.legend()
pl.grid(True)
'''


# Test #2: sandbox for new optimization
from scipy.optimize import minimize

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
    max_rate = 1000 * config.max_rate * config.drug_mg_ml / config.weight # in mcg/kg/min
    
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
                        bolus=None,
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
'''NOTES
This is working amazingly well. The next step is to make a class that handles realtime management. So it should first allow designing a plan like a walkup, but it should be able to be realtime-called and rapidly change course, specifically to hold at a given level. We'll achieve this by making a queue of upcoming instructions, which gets reset if user decides to at any point. And it'll dump very fast short-term instructions into that queue, then in the background prepare longer-term instructions and dump onto the back.
'''
t = TCI(age=age, weight=weight, height=height, sex=sex)
vals = []

def implement(t, rates, durs, vals):
    for r,d in zip(rates, durs):
        t.infuse(r)
        vals += collect_vals(t, dur=d)

rates, durs = go_to_target(t, 1.0, duration=5*60, new_target_travel_time=2*60)
implement(t, rates, durs, vals)

rates, durs = go_to_target(t, 1.1, duration=3*60)
implement(t, rates, durs, vals)

rates, durs = go_to_target(t, 1.2, duration=3*60)
implement(t, rates, durs, vals)

rates, durs = go_to_target(t, 1.3, duration=3*60)
implement(t, rates, durs, vals)

# now they LOR'd, so I dynamically say hey hold it here for 10 mins
rates, durs = go_to_target(t, t.level, duration=5*60)
implement(t, rates, durs, vals)

cp, ce = np.array(vals).T
pl.figure()
#pl.plot(np.arange(len(cp))/60, cp, label='cp', color='gold', ls='--')
pl.plot(np.arange(len(ce))/60, ce, label='ce', color='gold',)
pl.legend(); pl.grid(True)
##

t = TCI(age=age, weight=weight, height=height, sex=sex)
d,r = compute_bolus_to_reach_ce_2(t, 4.5)
print((d * r/60) * weight / 1000)
y = compute_bolus_to_reach_ce(t, 2.0)

vals = []
rates = [r,0]
durs = [d,60*5]
for r,d in zip(rates, durs):
    t.infuse(r)
    vals += collect_vals(t, dur=d)
##


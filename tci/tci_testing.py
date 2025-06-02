##
import fus_anes.config as config
from fus_anes.tci import TCI_Propofol as TCI
from fus_anes.tci.tci_util import compute_bolus_to_reach_ce, bolus_to_infusion, hold_at_level, simulate

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

# Test #2: ensuring the tools work
'''
t = TCI(age=age, weight=weight, height=height, sex=sex)
vals = []

# bolus to infusion
b = compute_bolus_to_reach_ce(t, 5.0, initial_guess=1.0*weight)
s, d = bolus_to_infusion(b)
t.infuse(d)
vals += collect_vals(t, dur=s)
t.infuse(0)
    
# infusion target
#rate = compute_infusion_rate_for_target(t, 3.0, 30.0)
#t.infuse(rate)
#vals += collect_vals(t, dur=30)
#t.infuse(0)
#vals += collect_vals(t, dur=10*60)

# hold at level
resolution = 15
steps = hold_at_level(t, 3.0, 30*60, resolution=resolution, initial_guess=75, foresight=4)
for step in steps:
    t.infuse(step)
    vals += collect_vals(t, dur=resolution)
t.infuse(0)
vals += collect_vals(t, dur=10*60)

cp, ce = np.array(vals).T
pl.figure()
pl.plot(np.arange(len(ce))/60, ce)
'''

# Test #3: checking infusion protocols
'''
initial_bolus_ce = 5.0
bfr = 0.0
target_vals = [3.0, 0.0]
target_durations = [60*30, 60*20]
tp = TCIProtocol(
                 model_params=dict(age=age,
                                   weight=weight,
                                   height=height,
                                   sex=sex),
                 initial_bolus_ce=initial_bolus_ce,
                 bolus_refractory_period=bfr, # secs
                 target_values=target_vals,
                 target_durations=target_durations,
                 resolution=15, # secs,
                 foresight=4, # secs
        )
tp.generate()
    
# check what instructions produce
tci = TCI(age=age, weight=weight, height=height, sex=sex)
vals = []
for _,ins in tp.instructions.iterrows():
    if ins.kind == 0:
        # bolus, given as infusion
        dur, rate = bolus_to_infusion(ins.dose)
        tci.infuse(rate)
        vals += collect_vals(tci, dur=dur)
        tci.infuse(0)
        # hold for specified duration
        d = ins.target_dur - dur
        vals += collect_vals(tci, d)

    elif ins.kind == 1:
        tci.infuse(ins.dose)
        vals += collect_vals(tci, ins.target_dur)

cp, ce = np.array(vals).T
#pl.figure()
pl.plot(np.arange(len(vals))/60, cp, color='grey', ls=':')
pl.plot(np.arange(len(vals))/60, ce, color='k', ls='-')
'''

# Test #4: sandbox for new optimization
from scipy.optimize import minimize

# ok good, now use this code here to create a new version of "compute rate to reach target in specified time"

def rate_for_target(target_levels,
                    target_durations,
                    tci_obj,
                    instruction_interval=15.0,
                    sim_resolution=1.0,
                    ):
    '''This is meant for targets at some degree of steady state. boluses are still best computed using compute_bolus_to_reach_ce

    Target level: ce goal
    Target time: secs from LAST target
    Instruction interval: secs between rate changes
    Sim resolution: time steps of sim (remember small is important because boluses can be so fast)
    '''
    max_rate = 1000 * config.max_rate * config.drug_mg_ml / config.weight # in mcg/kg/min
    
    # this is a special step that adds 1 extra interval on so it doesnt get lazy with final instruction
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

    ''' 
    # realistic check: checks if most extreme action would get you to goal (right now checks last target only)
    unrealistic = False
    last_target = evaluation_values[-1]
    if last_target < tci_obj.ce:
        end = simulate(tci_obj, infusion=0, dur=end_target_time)[-1]
        unrealistic = end > last_target + 0.05
    elif last_target > tci_obj.ce:
        end = simulate(tci_obj, infusion=max_rate, dur=end_target_time)[-1]
        unrealistic = end < last_target - 0.05
    if unrealistic:
        return None, None
    '''
    
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



t = TCI(age=age, weight=weight, height=height, sex=sex)

vals = []

# find a way to reach and then stay stable for a bit more time
target_durations = [60*3, 20, 20, 20]
target_levels = [0.2] * len(target_durations)
rates, durs = rate_for_target(target_levels, target_durations, t, instruction_interval=20.0, sim_resolution=1.0)
for r,d in zip(rates, durs):
    t.infuse(r)
    vals += collect_vals(t, dur=d)

# now stay stable for a bunch of time
target_durations = [15] * 24
target_levels = [0.2] * len(target_durations)
rates, durs = rate_for_target(target_levels, target_durations, t, instruction_interval=15.0, sim_resolution=1.0)
for r,d in zip(rates, durs):
    t.infuse(r)
    vals += collect_vals(t, dur=d)

# now reach new target and then stay stable for a bit more time
target_durations = [60*3, 20, 20, 20]
target_levels = [0.3] * len(target_durations)
rates, durs = rate_for_target(target_levels, target_durations, t, instruction_interval=20.0, sim_resolution=1.0)
for r,d in zip(rates, durs):
    t.infuse(r)
    vals += collect_vals(t, dur=d)

# now stay stable again
target_durations = [15] * 24
target_levels = [0.3] * len(target_durations)
rates, durs = rate_for_target(target_levels, target_durations, t, instruction_interval=15.0, sim_resolution=1.0)
for r,d in zip(rates, durs):
    t.infuse(r)
    vals += collect_vals(t, dur=d)

# now reach new target and then stay stable for a bit more time
target_durations = [60*3, 20, 20, 20]
target_levels = [0.1] * len(target_durations)
rates, durs = rate_for_target(target_levels, target_durations, t, instruction_interval=20.0, sim_resolution=1.0)
for r,d in zip(rates, durs):
    t.infuse(r)
    vals += collect_vals(t, dur=d)

# now stay stable again
target_durations = [15] * 24
target_levels = [0.1] * len(target_durations)
rates, durs = rate_for_target(target_levels, target_durations, t, instruction_interval=15.0, sim_resolution=1.0)
for r,d in zip(rates, durs):
    t.infuse(r)
    vals += collect_vals(t, dur=d)

cp, ce = np.array(vals).T
pl.figure()
#pl.plot(np.arange(len(cp))/60, cp, label='cp', color='gold', ls='--')
pl.plot(np.arange(len(ce))/60, ce, label='ce', color='gold',)
pl.legend()
pl.grid(True)
##


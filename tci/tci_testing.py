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

def rate_for_target(target_level,
                    target_time,
                    tci_obj,
                    instruction_interval=10.0,
                    sim_resolution=0.200,
                    stability_lookahead_max=60.0*5,
                    stability_weight=0.5):
    '''This is meant for targets at some degree of steady state. boluses are still best computed using compute_bolus_to_reach_ce

    Target level: ce goal
    Target time: secs from now
    Instruction interval: secs between rate changes
    Sim reslution: time steps of sim (remember small is important because boluses can be so fast)
    Stability lookahead_max: the MAX time (in seconds) at which to compute forward-looking stability
    Stability weight: higher means more emphasis on making curve flat at end; 0 means doesn't matter

    '''
    max_rate = 1000 * config.max_rate * config.drug_mg_ml / config.weight # in mcg/kg/min
    ttimes = np.arange(0, target_time+1, instruction_interval)
    if ttimes[-1] != target_time:
        ttimes = np.append(ttimes, target_time)
    durations = np.diff(ttimes)

    def simulation_error(rates, target, duration, tci_obj):
        # incorporate the lookahead step - 1 step into future where nothing gets changed
        rates = np.append(rates, rates[-1])
        duration = np.append(duration, stability_lookahead_max)
        nsamp = int(round(stability_lookahead_max / sim_resolution))

        traj = simulate(tci_obj, infusion=rates, dur=duration, bolus=None, sim_resolution=sim_resolution)
        reached = traj[-nsamp] # the end of the instruction set - everything after is the lookahead

        stability_error = np.sum((traj[-nsamp+1:] - reached)**2) / target # I normalize by target because it empirically seems to work better 

        error_target = np.abs(reached - target)
        error = error_target + stability_weight * stability_error

        return error

    # realistic check: checks if most extreme action would get you to goal
    unrealistic = False
    if target_level < tci_obj.ce:
        end = simulate(tci_obj, infusion=0, dur=target_time)[-1]
        unrealistic = end > target_level + 0.05
    elif target_level > tci_obj.ce:
        end = simulate(tci_obj, infusion=max_rate, dur=target_time)[-1]
        unrealistic = end < target_level - 0.05
    if unrealistic:
        return None, None
    
    initial_guess_rates = np.array([tci_obj.infusion_rate] * len(durations))
    opt = minimize(simulation_error,
                   initial_guess_rates,
                   method='L-BFGS-B',
                   args=(target_level, durations, tci_obj),
                   options={},
                   bounds=[(0, max_rate)])

    return opt.x, durations



t = TCI(age=age, weight=weight, height=height, sex=sex)

vals = []
t.bolus(30)

# this is how you'd jump over to a new target
rates, durs = rate_for_target(0.75, 60*4.0, t)
for r,d in zip(rates, durs):
    t.infuse(r)
    vals += collect_vals(t, dur=d)

# then keep it steady state in 2min chunks - this is the way i've seen works best, short intervals aiming for the end to be accurate, without any stability requirement
for _ in range(5):
    rates, durs = rate_for_target(0.75, 60*2.0, t, instruction_interval=20.0, sim_resolution=0.5, stability_weight=0.0)
    for r,d in zip(rates, durs):
        t.infuse(r)
        vals += collect_vals(t, dur=d)

vals += collect_vals(t, dur=10*60)

cp, ce = np.array(vals).T
pl.figure()
#pl.plot(np.arange(len(cp))/60, cp, label='cp', color='gold', ls='--')
pl.plot(np.arange(len(ce))/60, ce, label='ce', color='gold',)
pl.legend()
pl.grid(True)
##


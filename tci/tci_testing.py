##
import fus_anes.config as config

age = config.age
weight = config.weight # kg
height = config.height # cm
sex = config.sex

def collect_vals(t, dur=10*60):
    # dur in secs
    vals = []
    nsteps = int(round(dur * 10))
    for _ in range(nsteps):
        t.wait(0.1)
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


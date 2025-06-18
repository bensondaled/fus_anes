##
import fus_anes.config as config
from fus_anes.tci import TCI_Propofol as TCI
from fus_anes.tci.tci_util import bolus_to_infusion, simulate, compute_bolus_to_reach_ce, go_to_target, mlmin_mcgkgmin

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


## Test #1: comparing to shafer with specific arbitrary scenarios
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


## Test #2: bolus for ce
t = TCI(age=age, weight=weight, height=height, sex=sex)
d,r = compute_bolus_to_reach_ce(t, 3.75)
print((d * r/60) * weight / 1000)

vals = []
rates = [r,0]
durs = [d,60*5]
for r,d in zip(rates, durs):
    t.infuse(r)
    vals += collect_vals(t, dur=d)

cp, ce = np.array(vals).T
pl.figure()
pl.plot(np.arange(len(cp))/60, cp, label='cp', color='gold', ls='--')
pl.plot(np.arange(len(ce))/60, ce, label='ce', color='gold',)
pl.legend(); pl.grid(True)


## Test #3: fancy target reaching
t = TCI(age=age, weight=weight, height=height, sex=sex)
vals = []

def implement(t, rates, durs, vals):
    for r,d in zip(rates, durs):
        t.infuse(r)
        vals += collect_vals(t, dur=d)

rates, durs = go_to_target(t, 0.5, duration=4*60, new_target_travel_time=90.0)
implement(t, rates, durs, vals)

rates, durs = go_to_target(t, 1.0, duration=5*60, new_target_travel_time=2*60)
implement(t, rates, durs, vals)

rates, durs = go_to_target(t, 1.1, duration=3*60)
implement(t, rates, durs, vals)

rates, durs = go_to_target(t, 1.2, duration=3*60)
implement(t, rates, durs, vals)

rates, durs = go_to_target(t, 1.3, duration=3*60)
implement(t, rates, durs, vals)

# pretend now they reached our desired state, so I dynamically say hey hold it here for a while
rates, durs = go_to_target(t, t.level, duration=5*60)
implement(t, rates, durs, vals)

cp, ce = np.array(vals).T
pl.figure()
pl.plot(np.arange(len(cp))/60, cp, label='cp', color='gold', ls='--')
pl.plot(np.arange(len(ce))/60, ce, label='ce', color='gold',)
pl.legend(); pl.grid(True)

## SANDBOX: toying with making LiveTCI
##

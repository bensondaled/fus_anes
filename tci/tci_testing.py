##
import fus_anes.config as config
from fus_anes.tci import TCI_Propofol as TCI, LiveTCI
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

implement(t, [0], [10*60], vals)

rates, durs = go_to_target(t, 1.0, duration=4*60, new_target_travel_time=90.0)
implement(t, rates, durs, vals)
rates, durs = go_to_target(t, 1.0, duration=5*60)
implement(t, rates, durs, vals)
rates, durs = go_to_target(t, 1.0, duration=2.5*60)
implement(t, rates, durs, vals)

rates, durs = go_to_target(t, 2.0, duration=5*60, new_target_travel_time=60.0)
implement(t, rates, durs, vals)
rates, durs = go_to_target(t, 2.0, duration=6*60)
implement(t, rates, durs, vals)

rates, durs = go_to_target(t, 3.0, duration=5*60, new_target_travel_time=60.0)
implement(t, rates, durs, vals)
rates, durs = go_to_target(t, 3.0, duration=6*60)
implement(t, rates, durs, vals)

rates, durs = go_to_target(t, 4.0, duration=5*60, new_target_travel_time=60.0)
implement(t, rates, durs, vals)
rates, durs = go_to_target(t, 4.0, duration=6*60)
implement(t, rates, durs, vals)

rates, durs = go_to_target(t, 3.5, duration=5*60, new_target_travel_time=60.0)
implement(t, rates, durs, vals)
rates, durs = go_to_target(t, 3.5, duration=6*60)
implement(t, rates, durs, vals)

rates, durs = go_to_target(t, 2.5, duration=5*60, new_target_travel_time=60.0)
implement(t, rates, durs, vals)
rates, durs = go_to_target(t, 2.5, duration=6*60)
implement(t, rates, durs, vals)

rates, durs = go_to_target(t, 1.5, duration=5*60, new_target_travel_time=60.0)
implement(t, rates, durs, vals)
rates, durs = go_to_target(t, 1.5, duration=6*60)
implement(t, rates, durs, vals)

rates, durs = go_to_target(t, 0.5, duration=5*60, new_target_travel_time=60.0)
implement(t, rates, durs, vals)
rates, durs = go_to_target(t, 0.5, duration=6*60)
implement(t, rates, durs, vals)

implement(t, [0], [45*60], vals)

# pretend now they reached our desired state, so I dynamically say hey hold it here for a while
#rates, durs = go_to_target(t, t.level, duration=5*60)
#implement(t, rates, durs, vals)

cp, ce = np.array(vals).T
fig, ax = pl.subplots(figsize=(5.5, 3.3),
                      gridspec_kw=dict(bottom=0.2, top=0.95, right=0.99))
#ax.plot(np.arange(len(cp))/60, cp, label='cp', color='lightseagreen', ls='--')
ax.plot((np.arange(len(ce))/60)[::2], ce[::2], label='ce', color='slategrey', lw=6)
#ax.legend()
ax.grid(True)
ax.set_yticks([0,1,2,3,4])
ax.set_xticks([0, 30, 60, 90, 120])
ax.tick_params(labelsize=20)
ax.set_xlabel('Minutes', fontsize=20)
ax.set_ylabel('[propofol] (mcg/ml)', fontsize=20)

## Test 4: LiveTCI
lt = LiveTCI()
lt.goto(1.0)
pl.plot(*lt.history)
pl.plot(*lt.get_projection())
pl.plot(*lt.simulation(infusion=0))
##

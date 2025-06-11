##
import fus_anes.config as config
from fus_anes.tci import TCI_Propofol as TCI
from fus_anes.tci.tci_util import bolus_to_infusion, simulate, compute_bolus_to_reach_ce, go_to_target
from fus_anes.hardware import Pump
import time

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

# pretend now they reached our desired state, so I dynamically say hey hold it here for a while
rates, durs = go_to_target(t, t.level, duration=5*60)
implement(t, rates, durs, vals)

cp, ce = np.array(vals).T
pl.figure()
pl.plot(np.arange(len(cp))/60, cp, label='cp', color='gold', ls='--')
pl.plot(np.arange(len(ce))/60, ce, label='ce', color='gold',)
pl.legend(); pl.grid(True)

## SANDBOX: toying with making LiveTCI
import multiprocessing as mp
class LiveTCI(mp.Process):
    '''
    Class that simultaneously handles infusion of a drug by being the overseer of both the TCI logic and the pump instructions

    Much to work on here - primarily there's a few things this needs to do:
    - handle the logic of just being asked to bolus or infuse something
    - handle requests for simulated future
    - handle requests for reaching targets, including allowing for lagged calculations but realtime implementations
    '''
    def __init__(self, **tci_kw):
        super(LiveTCI, self).__init__()

        self.pump = Pump() # TODO: note we are making the pump object here now
        self.tci = TCI(**tci_kw)
        self.simulated = [] # each time user asks for anything from object, will update this so it has entire history of what happened to this TCI
        self.instruction_queue = mp.Queue() # stores what's planned next for the TCI
        
        self.kill_flag = mp.Value('b', 0)
        self._on = mp.Value('b', 0)
        self.start()

    @property
    def level(self):
        return self.tci.level

    def goto(self, target, **kw):
        kw['duration'] = kw.pop('duration', 3*60)
        kw['new_target_travel_time'] = kw.pop('new_target_travel_time', 1*60)
        # TODO: somehow send this goal to the inside of the main process to get computed in a separate thread on the updated tci object, then implemented. you might sort of need two ongoing threads, one that's handling computations for things to add to the main instruction queue, and another just constantly checking the main instruction queue and implementing whatever is due
        #rates, durs = go_to_target(t, target, **kw)
    
    def simulate(self):
        return

    def run(self):
        while not self.kill_flag.value:
            try:
                rate, dur = self.instruction_queue.get(block=False)
            except queue.Empty:
                pass
        
        self.pump.end()
        self._on = False
    
    def end(self):
        self.kill_flag.value = 1
        while self._on.value:
            time.sleep(0.010)
##

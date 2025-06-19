import copy
import numpy as np
from scipy.optimize import minimize
import multiprocessing as mp
import threading
import queue
import time
import bisect

import fus_anes.config as config
from fus_anes.hardware import Pump
from fus_anes.tci import TCI_Propofol, TCI_Ketamine
from fus_anes.util import now, now2

if config.drug.lower() == 'propofol':
    TCI = TCI_Propofol
elif config.drug.lower() == 'ketamine':
    TCI = TCI_Ketamine

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

    obj = copy.deepcopy(tci_obj)

    if sim_resolution is None:
        sim_resolution = obj.resolution

    if sim_resolution < obj.resolution:
        print('Adjusting sim resolution to be at least TCI resolution')
        sim_resolution = obj.resolution

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

class LiveTCI(mp.Process):
    '''
    Class that simultaneously handles infusion of a drug by being the overseer of both the TCI logic and the pump instructions

    Much to work on here - primarily there's a few things this needs to do:
    - handle the logic of just being asked to bolus or infuse something
    - handle requests for simulated future
    - handle requests for reaching targets, including allowing for lagged calculations but realtime implementations
    '''
    def __init__(self,
                 saver=None,
                 new_instruction_delay=10.000, # longer is safer but also means waiting more to do things
                 history_interval=2.0,
                 projection_duration=20*60,
                 projection_interval=5.0,
                 simulation_duration=20*60,
                 simulation_interval=5.0,
                 **tci_kw):
        super(LiveTCI, self).__init__()

        self.tci_kw = tci_kw
        self.new_instruction_delay = new_instruction_delay # the purpose of this is: when a brand new instruction is sent, there's often overhead getting it into the line and immediately implemented at this very moment. This would be fine, except that this instruction may have contingencies that follow it, for example a stop time meant to be exactly n milliseconds after the first move. If we just delay the start of the first move a bit, then both instructions can be processed and placed in the queue, and the first one will start on time thus making the interval between them more accurate.

        self.instruction_queue = mp.Queue() # (rate_in_mcgkgmin, absolute_now()_time_to_run_that_rate)
        self.user_request_queue = mp.Queue()
        
        self.level_request_flag = mp.Value('b', 0)
        self._level = mp.Value('d', 0.0)
        self.rate_request_flag = mp.Value('b', 0)
        self._rate = mp.Value('d', 0.0)
        
        self.history_interval = history_interval
        self.history_request_flag = mp.Value('b', 0)
        self._history = mp.Queue()

        self.projection_duration = projection_duration
        self.projection_interval = projection_interval
        self.projection_request_flag = mp.Value('b', 0)
        self._projection = mp.Queue()
        
        self.simulation_duration = simulation_duration
        self.simulation_interval = simulation_interval
        self.simulation_request_flag = mp.Value('b', 0)
        self.sim_infusion = mp.Value('d', 0.0)
        self._simulation = mp.Queue()
        
        self.export_request_flag = mp.Value('b', 0)
        self._export = mp.Queue()
        
        self.clear_instruction_queue_flag = mp.Value('b', 0)

        self.kill_flag = mp.Value('b', 0)
        self._on = mp.Value('b', 0)
        self.start()

    @property
    def level(self):
        self.level_request_flag.value = True
        while self.level_request_flag.value:
            time.sleep(0.025)
        return self._level.value
    
    @property
    def infusion_rate(self):
        self.rate_request_flag.value = True
        while self.rate_request_flag.value:
            time.sleep(0.025)
        return self._rate.value
    
    @property
    def history(self):
        self.history_request_flag.value = True
        hist, t0 = self._history.get(block=True)
        htime = np.arange(len(hist)) * self.history_interval + t0
        return htime, hist
    
    @property
    def projection(self):
        self.projection_request_flag.value = True
        proj = self._projection.get(block=True)
        ptime = np.arange(len(proj)) * self.projection_interval
        return ptime, proj
    
    def simulation(self, infusion=0.0):
        self.sim_infusion.value = infusion

        self.simulation_request_flag.value = True
        sim = self._simulation.get(block=True)
        stime = np.arange(len(sim)) * self.simulation_interval
        return stime, sim

    def bolus(self, dose=None, ce=None):
        if (dose==0 and ce==None) or (dose==None and ce==0):
            return

        self.clear_instruction_queue()
        self.user_request_queue.put(('bolus', [dose, ce]))

    def infuse(self, rate, dur=None, clear_queue=True):
        '''Accepts rate in mcg/kg/min
        '''
        if clear_queue:
            self.clear_instruction_queue()
        self.user_request_queue.put(('infuse', [rate, dur]))

    def goto(self, target, **kw):
        # TODO: I think the default behaviour here should be to go and to hold indefinitely, which means it needs a cascade of renewing goto commands. implement it in process_user_requests

        self.clear_instruction_queue()
        self.user_request_queue.put(('goto', [target, kw]))

    def keep_history(self):
        history = []
        last_update = now()
        while not self.kill_flag.value:

            # provide history if user requested it
            if self.history_request_flag.value:
                self._history.put([np.array(history), self.t0])
                self.history_request_flag.value = 0

            if now() - last_update >= self.history_interval:
                last_update = now()
                history.append(self.level)

    def clear_instruction_queue(self):
        self.clear_instruction_queue_flag.value = 1
        while self.clear_instruction_queue_flag.value:
            time.sleep(0.010)
    
    def process_user_requests(self):
        # The purpose of this is to be a parallel continuous ongoing process that handles not the implementation of direct commands to the pump or TCI, but the conversion of user requests into commands.
        
        while not self.kill_flag.value:
            try:
                kind, params = self.user_request_queue.get(block=False)

                if kind == 'bolus':
                    dose, ce = params

                    if ce is not None:
                        dose = compute_bolus_to_reach_ce(self.tci, ce)
                    if dose is None:
                        return

                    secs, rate = bolus_to_infusion(dose)
                    self.infuse(rate, dur=secs, clear_queue=False)

                elif kind == 'infuse':
                    rate, dur = params

                    previous_rate = self.tci.infusion_rate
                    start_time = now() #+ self.new_instruction_delay # leave out the delay here, because direct commands for infusions and boluses will otherwise be delayed
                    
                    # handle infusion start
                    instruction = (rate, start_time)
                    self.instruction_queue.put(instruction)

                    # handle infusion end if a duration was specified
                    if dur is not None:
                        stop_time = start_time + dur
                        return_rate = previous_rate
                        instruction = (return_rate, stop_time)
                        self.instruction_queue.put(instruction)

                elif kind == 'goto':
                    target, kw = params

                    kw['duration'] = kw.pop('duration', 4*60)
                    kw['new_target_travel_time'] = kw.pop('new_target_travel_time', 90.0)

                   
                    #self.infuse(self.tci.infusion_rate) # halt everything and keep infusing at current rate

                    start_time = now()

                    _, sim_obj = simulate(self.tci, [self.tci.infusion_rate], [self.new_instruction_delay], return_sim_object=True) # because the coming calculation may take some time, increment some simulated time from which to start the whole operation, so that when we have our computation complete and ready to implement, it is happening from the correct starting conditions
                    rates, durs = go_to_target(sim_obj, target, **kw)

                    cmd_time = start_time + self.new_instruction_delay
                    for rate, dur in zip(rates, durs):
                        instruction = (rate, cmd_time)
                        self.instruction_queue.put(instruction)
                        cmd_time += dur
                    
                    # now we implement the special logic of holding it there beyond the user's request
                    _, sim_obj = simulate(sim_obj, rates, durs, return_sim_object=True)
                    rates, durs = go_to_target(sim_obj, target, duration=5*60)
                    for rate, dur in zip(rates, durs):

                        if cmd_time < now(): # this is purely a failsafe: if somehow the logic took too long and was planning to submit an instruction that's already overdue, just push all upcoming instructions from this plan forward to start now. it's not ideal because it means the calculations that went into it were run on "early" TCI info, but it's better than running something super late and assuming it ran on time
                            cmd_time = now()

                        instruction = (rate, cmd_time)
                        self.instruction_queue.put(instruction)
                        cmd_time += dur

                        # notice how the last duration is not incorporated into the instructions, ie it passes on info about when to start, but without any other steps, it will just continue that rate indefinitely

            except queue.Empty:
                pass

    def project(self, instructions):
        if len(instructions) != 0:
            i_rates, i_starts = zip(*instructions)
        else:
            i_rates = []
            i_starts = []

        rates = [self.tci.infusion_rate] + np.array(i_rates).tolist()

        starts = np.append(now(), i_starts)
        durs = np.diff(starts)
        remaining_time = self.projection_duration - np.sum(durs)
        durs = np.append(durs, remaining_time)
        durs[0] = max(durs[0], 0.002)
        durs = np.array(durs).tolist()

        proj = simulate(self.tci, rates, durs, sim_resolution=self.projection_interval, report_resolution=None)

        self._projection.put(proj)
    
    def simulate(self):
        # TODO incorporate bolus
        infusion = self.sim_infusion.value

        rates, durs = [infusion], [self.simulation_duration]
        sim = simulate(self.tci, rates, durs,
                       sim_resolution=self.simulation_interval,
                       report_resolution=None)

        self._simulation.put(sim)

    def run(self):
        self.pump = Pump()
        self.tci = TCI(**self.tci_kw)
        self.t0 = now()

        threading.Thread(target=self.process_user_requests, daemon=True).start()
        threading.Thread(target=self.keep_history, daemon=True).start()

        # This loop handles all direct instructions in a continuous ongoing manner. These are all uniformly specified by a rate and an absolute time to run that rate. To get instructions into that line, other threads can add to the instruction_queue, and this loop handles the rest (sorting and implementing them).
        queued_instructions = []
        abs_start_time = now()
        last_tci_cmd_time = now()
        while not self.kill_flag.value:

            # if requested, clear the instruction queue
            if self.clear_instruction_queue_flag.value:
                queued_instructions = []
                self.clear_instruction_queue_flag.value = 0

            # grab any new items and throw them into the waiting list
            try:
                rate, ts = self.instruction_queue.get(block=False)
                # add anything new from the queue in its proper position in the working instruction list
                bisect.insort(queued_instructions, (rate, ts), key=lambda inst: inst[1])
            except queue.Empty:
                pass
            
            # provide level if user requested it
            if self.level_request_flag.value:
                self.tci.wait(now() - last_tci_cmd_time)
                last_tci_cmd_time = now()
                self._level.value = self.tci.level
                self.level_request_flag.value = 0
            
            # and rate
            if self.rate_request_flag.value:
                self._rate.value = self.tci.infusion_rate
                self.rate_request_flag.value = 0

            # provide projection if user requested it
            if self.projection_request_flag.value:
                threading.Thread(target=self.project, args=(queued_instructions.copy(),), daemon=True).start()
                self.projection_request_flag.value = 0

            # provide simulation if user requested it
            if self.simulation_request_flag.value:
                threading.Thread(target=self.simulate, daemon=True).start()
                self.simulation_request_flag.value = 0
            
            # export if user requested it
            if self.export_request_flag.value:
                threading.Thread(target=self.run_export, daemon=True).start()
                self.export_request_flag.value = 0

            
            # then run the next step on the waiting list
            if len(queued_instructions) == 0:
                continue
            next_due = queued_instructions[0][1]
            if now() < next_due:
                continue

            #print('\n'.join([str((r, t-abs_start_time)) for r,t in queued_instructions]))
            

            rate, ts = queued_instructions.pop(0)
            print(f'Running instruction: rate={rate}, at time {ts} (delay = {1000*(now()-ts):0.2f}ms)')

            # handle the pump
            mlmin = mlmin_mcgkgmin(mcgkgmin=rate)
            self.pump.infuse(mlmin)

            # handle the TCI object
            self.tci.wait(now() - last_tci_cmd_time)
            self.tci.infuse(rate)
            last_tci_cmd_time = now()

            # handle the saver: TODO

        
        self.pump.end()
        self._on = False

    def run_export(self):
        self._export.put(self.tci.export())

    def export(self):
        self.export_request_flag.value = 1
        return self._export.get(block=True)

    def end(self):
        self.kill_flag.value = 1
        while self._on.value:
            time.sleep(0.010)

if __name__ == '__main__':
    from fus_anes.tci.tci_util import LiveTCI
    lt = LiveTCI()
    lt.goto(0.4)

    # and then when you want to try stuff:
    pl.plot(*lt.history)

    pl.plot(*lt.projection)

    pl.plot(*lt.simulation(infusion=0))

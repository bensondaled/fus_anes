import os
import json
import copy
import multiprocessing as mp
import pandas as pd
import queue
import threading
import logging
import time
from datetime import datetime as dtm

from fus_anes.hardware import EEG, Pump, Camera, Capnostream
from fus_anes.util import Saver, multitaper_spectrogram, now, now2
from fus_anes.tci import TCI, simulate, TCIProtocol, hold_at_level, bolus_to_infusion
import fus_anes.config as config

class Session():
    def __init__(self):
        sid = dtm.now()
        self.name = int(sid.strftime('%Y%m%d%H%M%S'))
        self.pretty_name = sid.strftime('%Y-%m-%d %H:%M:%S')
        self.tech_name = sid.strftime(f'%Y-%m-%d_%H-%M-%S_subject-{config.subject_id}')
        self.data_filename = f'{self.tech_name}.h5'
        self.data_file = os.path.join(config.data_path, self.data_filename)

        self.boluses = []
        self.infusions = []
        self.tci_vals = []
        self.simulated_tci_vals = []
        self.simulation_idx = 0
        self.prot_sim_vals = []
        self.prot_sim_idx = 0
        self.is_bolusing = False # False vs a list of [start_time, total_duration, pre_bolus_infusion_rate, backup_event_queue_list]

        self.markers = []

        self.running = False
        self.completed = False

    def run(self):
        self.error_queue = mp.Queue()
        self.saver = Saver(session_id=self.name, data_file=self.data_file, session_obj=self, error_queue=self.error_queue)
        self.tci = TCI(prior_tcm=self.get_prior_tcm())
        self.pump = Pump(error_queue=self.error_queue, saver=self.saver)
        self.cam = Camera(self.tech_name, error_queue=self.error_queue)
        self.capnostream = Capnostream(saver_obj_buffer=self.saver.buffer, error_queue=self.error_queue)
        self.eeg = EEG(saver_obj_buffer=self.saver.buffer, error_queue=self.error_queue)
        trun = now()
        self.tci_vals.append([trun, self.tci.level, trun])
        self.running = trun
        self.eeg.start_processing()

    def get_prior_tcm(self):
        candidates = sorted([os.path.join(config.data_path, f) for f in os.listdir(config.data_path) if f.endswith(f'_subject-{config.subject_id}.h5') and f!=self.data_filename])[::-1]
        for cand in candidates:
            with pd.HDFStore(cand, 'r') as h:
                if 'tci_end' in h:
                    try:
                        te = h['tci_end']
                        ts = te.index[0]
                        te = json.loads(te.iloc[0].tcm)
                        te.update(ts=ts)
                        logging.info(f'Initializing with prior TCI model from {cand}')
                        return te
                    except:
                        logging.warning(f'Failed to load meaningful prior TCI model from {cand}')
                        pass
        logging.info(f'No prior TCI found for this subject.')
        return None

    def add_marker(self, info):
        ts, txt = info
        self.markers.append([ts, txt])
        self.saver.write('markers', dict(t=ts, text=f'{txt:<100}'))

    def give_bolus(self, info):
        if config.TESTING:
            logging.info('Delivering bolus:', info)

        if self.is_bolusing is not False:
            logging.warning('Cannot bolus while bolusing, bolus canceled.')
            return

        t, dose = info
        if dose == 0.0:
            return
        if dose > config.max_bolus:
            dose = config.max_bolus
            logging.warning(f'Reduced bolus to maximum which is {config.max_bolus}')
        
        self.boluses.append((t, dose))

        secs_required, mcg_kg_min = bolus_to_infusion(dose)

        #self.tci.bolus(dose) # we no longer use this since boluses are not instantly delivered, instead we run a temporary fast infusion; but we save it as a bolus here in terms of logging action taken
        self.saver.write('tci', dict(kind='bolus', dose=float(dose), mlmin=float(config.bolus_rate), t=t, duration=float(secs_required)))
        
        #logging.info(f'Bolusing, will give {dose}mg which takes {secs_required}secs')
        self.is_bolusing = [now(), secs_required, self.pump.current_infusion_rate, []]

        self.pump.infuse(config.bolus_rate)
        self.compute_tci_point()
        self.tci.infuse(mcg_kg_min)

        threading.Thread(target=self.manage_bolus).start()

    def manage_bolus(self):
        start, dur, restore, queue = self.is_bolusing
        while now() - start <= dur:
            time.sleep(0.010)

        #self.pump.infuse(restore)
        # rather than backend push the original rate, register it as a new infusion so it can be visually tracked and saved as such
        self.give_infusion([now(), restore], force=True, mlmin=True)

        for kind, info in queue:
            if kind == 'infusion':
                info[0] = now() # record real time of infusion change which is now, rather then when requested & queued
                self.give_infusion(info, force=True)
            time.sleep(0.250)

        self.is_bolusing = False
    
    def give_infusion(self, info, force=False, mlmin=False):
        # given in mcg/kg/min (unless mlmin==True)
        
        if config.TESTING:
            logging.info('Delivering infusion:', info)
    
        if (not force) and (self.is_bolusing is not False):
            self.is_bolusing[-1].append(('infusion', info))
            return

        t, dose = info # dose in mcg/kg/min
        if mlmin:
            dose_mlmin = dose
            dose_mgmin = dose_mlmin * 10.0
            dose_mcgmin = dose_mgmin * 1000.0
            dose_mcgkgmin = dose_mcgmin / config.weight
        else: # mcg/kg/min given
            dose_mcgkgmin = dose
            dose_mgmin = dose_mcgkgmin * config.weight / 1000
            dose_mlmin = dose_mgmin / 10.0 # bc propofol 10mg/ml
        
        if dose_mlmin > config.max_rate:
            dose_mlmin = config.max_rate
            dose_mgmin = dose_mlmin * 10.0
            dose_mcgkgmin = dose_mgmin * 1000.0 / config.weight # mcg/kg/min
            logging.warning(f'Reduced infusion rate to maximum which is {config.max_rate}ml/min ({dose} mcg/kg/min)')
            
        if 0 < dose_mlmin <= config.min_rate:
            dose_mlmin = 0
            dose_mgmin = 0
            dose_mcgkgmin = 0 # mcg/kg/min
            logging.warning(f'Requested pump rate below minimum, defaulting to 0.')
        
        self.infusions.append((t, dose_mcgkgmin))

        self.compute_tci_point()     

        self.pump.infuse(dose_mlmin)
        self.tci.infuse(dose_mcgkgmin)
        self.saver.write('tci', dict(kind='infusion', dose=float(dose_mcgkgmin), mlmin=float(dose_mlmin), t=t, duration=-1.0))

    def simulate_tci(self, bolus, infusion, duration=config.tci_sim_duration):
        self.simulation_idx = len(self.tci_vals)-1

        bolus = bolus_to_infusion(bolus)

        vals = simulate(self.tci, bolus, infusion, duration)
        self.simulated_tci_vals = vals
        return self.simulation_idx, self.simulated_tci_vals

    def compute_tci_point(self):
        t0 = self.tci_vals[-1][0]
        t = now()
        dt = t-t0
        if dt == 0:
            return
        self.tci.wait(dt)
        
        self.tci_vals.append([t, self.tci.level, t0+dt]) # true time last calculated, level, rounded time last calculated [meaningless ever since I improved resolution]; the "t" is the true time it was called, used to ensure we don't accumulate lag, whereas t0+dt is the time in the tci's universe which corresponds directly to the stored level
    
    def get_tci_curve(self):
        time = [t[2] for t in self.tci_vals] # uses the rounded times when returning curve because those align with the values calculated
        vals = [t[1] for t in self.tci_vals]
        return time, vals

    def retrieve_errors(self):
        errs = []
        while True:
            try:
                err = self.error_queue.get(block=False)
                errs.append(err)
            except queue.Empty:
                break
        return errs

    def reset_cam(self):
        del self.cam
        self.cam = Camera(f'{self.tech_name}_reset_{now()}', error_queue=self.error_queue)

    def reset_capnostream(self):
        del self.capnostream
        self.capnostream = Capnostream(saver_obj_buffer=self.saver.buffer, error_queue=self.error_queue)

    def end(self):
        self.running = False
        self.saver.write('tci_end', dict(tcm=json.dumps(self.tci.export())))
        to_end = [self.eeg, self.capnostream, self.saver, self.pump, self.cam]
        for te in to_end:
            try:
                te.end()
            except:
                logging.error(f'Failed to properly end {te}.')
            time.sleep(0.100)
        self.completed = True

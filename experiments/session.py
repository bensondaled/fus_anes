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

from fus_anes.hardware import EEG, Pump, Camera, Capnostream, Microphone
from fus_anes.verbal_instructions import SqueezeInstructions, BaselineEyes
from fus_anes.util import Saver, multitaper_spectrogram, now, now2
from fus_anes.tci import LiveTCI
import fus_anes.config as config

class Session():
    def __init__(self):
        sid = dtm.now()
        self.name = int(sid.strftime('%Y%m%d%H%M%S'))
        self.pretty_name = sid.strftime('%Y-%m-%d %H:%M:%S')
        self.tech_name = sid.strftime(f'%Y-%m-%d_%H-%M-%S_subject-{config.subject_id}')
        self.data_filename = f'{self.tech_name}.h5'
        self.data_file = os.path.join(config.data_path, self.data_filename)

        self.markers = []

        self.running = False
        self.completed = False

    def run(self):
        self.error_queue = mp.Queue()
        self.saver = Saver(session_id=self.name, data_file=self.data_file, session_obj=self, error_queue=self.error_queue)
        self.tci = LiveTCI(prior_tcm=self.get_prior_tcm())
        self.squeeze = None
        self.baseline_eyes = None
        self.pump = Pump(error_queue=self.error_queue, saver=self.saver)
        self.cam = Camera(self.tech_name, error_queue=self.error_queue)
        self.mic = Microphone(self.tech_name, error_queue=self.error_queue)
        self.capnostream = Capnostream(saver_obj_buffer=self.saver.buffer, error_queue=self.error_queue)
        self.eeg = EEG(saver_obj_buffer=self.saver.buffer, error_queue=self.error_queue)
        trun = now()
        self.running = trun
        self.eeg.start_processing()
    
    def toggle_baseline(self):
        if self.baseline_eyes is None:
            self.baseline_eyes = BaselineEyes()
            self.baseline_eyes.play()
        elif self.baseline_eyes.playing == False: # finished playing
            self.baseline_eyes.end()
            self.baseline_eyes = None

    def toggle_squeeze(self):
        if self.squeeze is None:
            self.squeeze = SqueezeInstructions()
            self.squeeze.play()
        else:
            self.squeeze.end()
            self.squeeze = None

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
        
        # TODO: implement bolus

    def give_infusion(self, info, force=False, mlmin=False):
        # given in mcg/kg/min (unless mlmin==True)
        
        if config.TESTING:
            logging.info('Delivering infusion:', info)
        
        # TODO: implement infusion


    def retrieve_errors(self):
        errs = []
        while True:
            try:
                err = self.error_queue.get(block=False)
                errs.append(err)
            except queue.Empty:
                break
        return errs

    def get_code_txt(self):
        py_files = [os.path.join(d,f) for d,_,fs in os.walk(os.getcwd()) for f in fs if f.endswith('.py') and not f.startswith('__')]
        code = {}
        for pf in py_files:
            with open(pf, 'r') as f:
                code[pf] = f.read()
        return json.dumps(code)

    def end(self):
        self.running = False
        self.saver.write('tci_end', dict(tcm=json.dumps(self.tci.export())))
        to_end = [self.eeg, self.capnostream, self.saver, self.pump, self.cam, self.mic, self.tci, self.baseline_eyes, self.squeeze]
        for te in to_end:
            print(te)
            if te is None:
                continue
            try:
                te.end()
            except:
                logging.error(f'Failed to properly end {te}.')
            time.sleep(0.100)
        self.completed = True

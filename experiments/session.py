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

from fus_anes.hardware import EEG, Pump, Camera, Microphone
from fus_anes.audio import SqueezeInstructions, BaselineEyes, Oddball, Chirp, end_audio
from fus_anes.audio.generate_squeeze_instruction_audio import make as make_squeeze_audio
from fus_anes.util import Saver, multitaper_spectrogram, now, save
from fus_anes.tci import LiveTCI
import fus_anes.config as config

class Session():
    def __init__(self):
        sid = dtm.now()
        self.name = int(sid.strftime('%Y%m%d%H%M%S'))
        self.pretty_name = sid.strftime('%Y-%m-%d %H:%M:%S')
        self.tech_name = sid.strftime(f'%Y-%m-%d_%H-%M-%S_subject-{config.subject_id}')
        self.data_filename = f'{self.tech_name}.h5'
        os.makedirs(config.data_path, exist_ok=True)
        self.data_file = os.path.join(config.data_path, self.data_filename)

        self.markers = []

        self.running = False
        self.completed = False

    def run(self):
        self.error_queue = mp.Queue()
        self.saver = Saver(session_id=self.name, data_file=self.data_file, session_obj=self, error_queue=self.error_queue)
        self.tci = LiveTCI(prior_tcm=self.get_prior_tcm(), error_queue=self.error_queue, saver_buffer=self.saver.buffer)
        self.squeeze = None
        make_squeeze_audio()
        self.baseline_eyes = None
        self.oddball = None
        self.chirp = None
        self.cam = Camera(self.tech_name, error_queue=self.error_queue)
        self.mic = Microphone(self.tech_name, error_queue=self.error_queue)
        self.eeg = EEG(saver_obj_buffer=self.saver.buffer, error_queue=self.error_queue)

        trun = now(minimal=True)
        self.running = trun
        self.eeg.start_processing()
     
    def toggle_baseline(self):
        self._run_auditory_process(cls=BaselineEyes, obj_name='baseline_eyes')
    def toggle_squeeze(self):
        self._run_auditory_process(cls=SqueezeInstructions, obj_name='squeeze')
    def toggle_oddball(self):
        self._run_auditory_process(cls=Oddball, obj_name='oddball')
    def toggle_chirp(self):
        self._run_auditory_process(cls=Chirp, obj_name='chirp')

    def _run_auditory_process(self, cls, obj_name):
        if getattr(self, obj_name) is None:
            obj = cls(saver_buffer=self.saver.buffer)
            setattr(self, obj_name, obj)
            obj.play()
            threading.Thread(target=self._nullify_when_done_playing, args=(obj_name,), daemon=True).start()

            t = now(minimal=True)
            self.add_marker([t, f'{obj_name} start'])
        else:
            obj = getattr(self, obj_name)
            obj.end()
            setattr(self, obj_name, None)
            
            t = now(minimal=True)
            self.add_marker([t, f'{obj_name} ended'])

    def _nullify_when_done_playing(self, obj_name):
        obj = getattr(self, obj_name)
        if obj is None:
            return
        while obj.playing.value:
            time.sleep(2.0)
        obj.end()
        setattr(self, obj_name, None)
        t = now(minimal=True)
        self.add_marker([t, f'{obj_name} complete'])

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
        save('markers', dict(t=ts, text=f'{txt:<100}'), self.saver.buffer)

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
        save('tci_end', dict(tcm=json.dumps(self.tci.export())), self.saver.buffer)
        to_end = [self.eeg, self.saver, self.cam, self.mic, self.tci, self.baseline_eyes, self.squeeze, self.oddball, self.chirp]
        for te in to_end:
            logging.info(f'Ending {te}')
            if te is None:
                continue
            try:
                te.end()
            except:
                logging.error(f'Failed to properly end {te}.')
            time.sleep(0.100)
        end_audio()
        self.completed = True

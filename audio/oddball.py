import os
import threading
import time
import numpy as np
import multiprocessing as mp

import fus_anes.config as config
from fus_anes.util import save

import sounddevice as sd
import soundfile as sf

sd.default.device = config.audio_in_ch_out_ch

if config.THREADS_ONLY:
    mproc = threading.Thread
else:
    mproc = mp.Process

class Oddball(mproc):
    def __init__(self, saver_buffer=None,
                 standard_tone_name='standard_tone.wav',
                 deviant_tone_name='deviant_tone.wav',):
        super(Oddball, self).__init__()

        audio_path = os.path.join(config.oddball_audio_path)
        self.standard_file = os.path.join(audio_path, standard_tone_name)
        self.deviant_file = os.path.join(audio_path, deviant_tone_name)

        self.saver_buffer = saver_buffer
        
        self.is_playing = mp.Value('b', 0)
        self.kill_flag = mp.Value('b', 0)

    def play(self):
        save('oddball', dict(event='P'), self.saver_buffer)
        self.start()

    def run(self):
        duration_min = config.oddball_duration_min
        isi_ms = config.oddball_isi_ms
        oddball_ratio = config.oddball_deviant_ratio

        standard_data, fs = sf.read(self.standard_file, dtype='float32')
        deviant_data, _ = sf.read(self.deviant_file, dtype='float32') 

        tone_duration_ms = int(len(standard_data) / fs * 1000)
        silence_duration_ms = isi_ms - tone_duration_ms
        silence_samples = int(silence_duration_ms / 1000 * fs)

        total_stimuli = int((duration_min * 60 * 1000) / isi_ms)
        n_deviants = int(total_stimuli * oddball_ratio)
        n_standards = total_stimuli - n_deviants

        sequence = np.array(['s'] * n_standards + ['d'] * n_deviants)
        n_allowed_consecutive = 0
        max_iters = 100000
        n_iters = 0
        while np.sum((sequence[:-1]=='d') & (sequence[1:]=='d')) > n_allowed_consecutive:
            np.random.shuffle(sequence)
            n_iters += 1
            if n_iters > max_iters:
                n_iters = 0
                n_allowed_consecutive += 1
        print(f'Allowed {n_allowed_consecutive} consecutive oddballs')
        sequence = np.append(np.array(['s'] * config.oddball_n_standard_start),  sequence)

        wait_ms = isi_ms - tone_duration_ms
        wait_sec = wait_ms / 1000.0
        self.is_playing.value = 1

        for i, stim_type in enumerate(sequence):
            if self.kill_flag.value:
                break

            if stim_type == 's':
                data = standard_data
            elif stim_type == 'd':
                data = deviant_data

            save('oddball', dict(event=stim_type), self.saver_buffer)
            sd.play(data, fs)
            sd.wait()
            time.sleep(wait_sec)

        self.is_playing.value = 0

    def end(self):
        self.kill_flag.value = True
        #while self.is_playing.value:
        #    time.sleep(0.025)

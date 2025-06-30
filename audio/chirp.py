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

class Chirp(mproc):
    def __init__(self, saver_buffer=None,
                 ):
        super(Chirp, self).__init__()

        self.chirp_file = config.chirp_audio_path
        self.chirp_white_file = config.chirp_white_audio_path

        self.saver_buffer = saver_buffer
        
        self.is_playing = mp.Value('b', 0)
        self.kill_flag = mp.Value('b', 0)

    def play(self):
        save('chirp', dict(event='P'), self.saver_buffer)
        self.start()

    def run(self):

        chirp_data, fs = sf.read(self.chirp_file, dtype='float32')
        white_data, _ = sf.read(self.chirp_white_file, dtype='float32')
        tone_duration_ms = int(len(chirp_data) / fs * 1000)
        
        n_reps = config.chirp_n_tones + config.chirp_n_start
        isi_ms = config.chirp_isi_ms
        
        isis = np.random.choice(np.arange(isi_ms[0], isi_ms[1]+1, 100), size=n_reps)
        wait_ms = isis - tone_duration_ms
        wait_sec = wait_ms / 1000.0

        n_ctrl = int(config.chirp_ctl_rate * config.chirp_n_tones)
        sequence = np.array(['c']*(config.chirp_n_tones-n_ctrl) + ['w']*n_ctrl)
        np.random.shuffle(sequence)
        sequence = np.append(np.array(['c']*config.chirp_n_start), sequence)

        self.is_playing.value = 1

        for seq,wait in zip(sequence, wait_sec):
            if self.kill_flag.value:
                break

            if seq == 'c':
                data = chirp_data
            elif seq == 'w':
                data = white_data

            save('chirp', dict(event=seq), self.saver_buffer)
            sd.play(data, fs)
            sd.wait()
            time.sleep(wait)

        self.is_playing.value = 0

    def end(self):
        self.kill_flag.value = True
        #while self.is_playing.value:
        #    time.sleep(0.025)

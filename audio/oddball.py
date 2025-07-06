import os
import threading
import time
import numpy as np
import multiprocessing as mp

import fus_anes.config as config
from fus_anes.util import save, now
from .audio_util import play_tone_precisely

import sounddevice as sd
import soundfile as sf

if config.audio_backend == 'ptb':
    from psychopy import sound

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
        
        self.playing = mp.Value('b', 1)
        self.kill_flag = mp.Value('b', 0)

    def play(self):
        save('oddball', dict(event='P', onset_ts=0.0, dummy=0.0), self.saver_buffer)
        self.start()

    def run(self):
        n_tones = config.oddball_n_tones
        isi_ms = config.oddball_isi_ms
        oddball_ratio = config.oddball_deviant_ratio
        
        if config.audio_backend == 'sounddevice':
            standard_data, fs = sf.read(self.standard_file, dtype='float32')
            deviant_data, _ = sf.read(self.deviant_file, dtype='float32') 
        elif config.audio_backend == 'ptb':
            standard_data = sound.Sound(self.standard_file)
            deviant_data = sound.Sound(self.deviant_file) 
            fs = 44100

        tone_duration_ms = int(len(standard_data) / fs * 1000)

        n_deviants = int(n_tones * oddball_ratio)
        n_standards = n_tones - n_deviants

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

        # --- play
        for i, stim_type in enumerate(sequence):
            if self.kill_flag.value:
                break

            if stim_type == 's':
                data = standard_data
            elif stim_type == 'd':
                data = deviant_data

            #sd.play(data, fs)
            #sd.wait()
            dummy = now(minimal=True)
            playtime = play_tone_precisely(data, fs)
            save('oddball', dict(event=stim_type, onset_ts=playtime, dummy=dummy), self.saver_buffer)
            
            _isi = np.random.randint(*isi_ms)
            wait_ms = _isi - 1000*(now(minimal=True)-playtime)
            print(wait_ms, playtime, now(minimal=True))
            if wait_ms > 0:
                time.sleep(wait_ms / 1000.0)

        self.playing.value = 0

    def end(self):
        self.kill_flag.value = True
        while self.playing.value:
            time.sleep(0.100)

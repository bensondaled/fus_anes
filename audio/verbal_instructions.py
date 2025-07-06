##
import os
import threading
import time
import numpy as np
import multiprocessing as mp

import fus_anes.config as config
from fus_anes.util import save
from .audio_util import play_tone_precisely

import sounddevice as sd
import soundfile as sf

if config.audio_backend == 'ptb':
    from psychopy import sound

sd.default.device = config.audio_in_ch_out_ch

def pad_str(s):
    # for saving
    min_length = 50
    return s.ljust(min_length)

if config.THREADS_ONLY:
    mproc = threading.Thread
else:
    mproc = mp.Process

class SqueezeInstructions(mproc):
    def __init__(self, with_nums=True, saver_buffer=None):
        super(SqueezeInstructions, self).__init__()
        self.name = config.name.lower()
        self.audio_path = os.path.join(config.verbal_instructions_path, self.name)
        self.interval = config.verbal_instruction_interval
        self.with_nums = with_nums
        
        self.saver_buffer = saver_buffer
        
        self.playing = mp.Value('b', 1)
        self.kill_flag = mp.Value('b', 0)

    def get_clip(self, i=None):
        suffix = f'_{i:0.0f}' if i is not None else ''
        filename = f'{self.name}{suffix}.mp3'
        path = os.path.join(self.audio_path, filename)
        if not os.path.exists(path):
            return None
        return path

    def play(self):
        save('squeeze', dict(event='play', isi=-1.0), self.saver_buffer)
        self.start()

    def run(self):
        idx = 1
        while not self.kill_flag.value:
            if self.with_nums:
                clip = self.get_clip(idx)
                if clip is None:
                    idx = 1
                    continue
            else:
                clip = self.get_clip(None)

            if config.audio_backend == 'sounddevice':
                data, samplerate = sf.read(clip)
            elif config.audio_backend == 'ptb':
                data = sound.Sound(clip)
                samplerate = 44100

            playtime = play_tone_precisely(data, samplerate)
            isi_ms = np.random.randint(self.interval[0]*1000, self.interval[1]*1000) # NOTE this ISI is from END of instruction unlike other auditory tasks
            save('squeeze', dict(event=os.path.split(clip)[-1], onset_ts=playtime, isi=isi_ms), self.saver_buffer)
            if isi_ms > 0:
                time.sleep(isi_ms / 1000.0)

            idx += 1
        self.playing.value = 0

    def end(self):
        self.kill_flag.value = True
        while self.playing.value:
            time.sleep(0.100)

class BaselineEyes(mproc):
    def __init__(self, names=['closed.mp3', 'open.mp3'], n_reps=2, dur=60.0,
                 saver_buffer=None):
        super(BaselineEyes, self).__init__()
        self.clip_paths = [os.path.join(config.baseline_audio_path, name) for name in names]
        self.n_reps = n_reps
        self.dur = dur
        self.saver_buffer = saver_buffer
        self.playing = mp.Value('b', 1)
        self.kill_flag = mp.Value('b', 0)
    
    def play(self):
        save('bl_eyes', dict(event='play'), self.saver_buffer)
        self.start()

    def run(self):
        paths = np.repeat(self.clip_paths, self.n_reps)
        np.random.shuffle(paths)

        for path in paths:
            if self.kill_flag.value:
                break
            data, samplerate = sf.read(path)
            save('bl_eyes', dict(event=os.path.split(path)[-1]), self.saver_buffer)
            sd.play(data, samplerate)
            sd.wait()
            time.sleep(self.dur)
        self.playing.value = 0

    def end(self):
        self.kill_flag.value = 1
        while self.playing.value:
            time.sleep(0.100)

if __name__ == '__main__':
    vi = SqueezeInstructions()
    vi.play()
    #vi.end()

    bl = BaselineEyes()
    bl.play()
##

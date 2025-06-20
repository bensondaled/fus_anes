##
import os
import threading
import time
import numpy as np
import multiprocessing as mp

import fus_anes.config as config

from psychopy.sound import Sound
from psychopy.core import wait
#import psychtoolbox as ptb

# TODO: incorporate saver

if config.THREADS_ONLY:
    mproc = threading.Thread
else:
    mproc = mp.Process

class SqueezeInstructions(mproc):
    def __init__(self, with_nums=True):
        super(SqueezeInstructions, self).__init__()
        self.name = config.name.lower()
        self.audio_path = os.path.join(config.verbal_instructions_path, self.name)
        self.interval = config.verbal_instruction_interval
        self.with_nums = with_nums
        
        self.is_playing = mp.Value('b', 0)
        self.kill_flag = mp.Value('b', 0)

    def get_clip(self, i=None):
        suffix = f'_{i:0.0f}' if i is not None else ''
        filename = f'{self.name}{suffix}.mp3'
        path = os.path.join(self.audio_path, filename)
        if not os.path.exists(path):
            return None
        return path

    def play(self):
        self.start()

    def run(self):
        self.is_playing.value = 1
        idx = 1
        while not self.kill_flag.value:
            if self.with_nums:
                clip = self.get_clip(idx)
                if clip is None:
                    idx = 1
                    continue
            else:
                clip = self.get_clip(None)

            s = Sound(clip)
            dur = s.getDuration()
            
            isi = self.interval[0] + np.random.normal(*self.interval[1])

            s.play() # To do: consider scheduling this play so its timing is more accurate
            wait(dur)

            wait(isi)
            idx += 1
        self.is_playing.value = 0

    def end(self):
        self.kill_flag.value = True
        #while self.is_playing.value:
        #    time.sleep(0.025)

class BaselineEyes(mproc):
    def __init__(self, names=['closed.mp3', 'open.mp3'], n_reps=2, dur=60.0):
        super(BaselineEyes, self).__init__()
        self.clip_paths = [os.path.join(config.baseline_audio_path, name) for name in names]
        self.n_reps = n_reps
        self.dur = dur
        self.playing = mp.Value('b', 0)
        self.kill_flag = mp.Value('b', 0)
    
    def play(self):
        self.start()

    def run(self):
        self.playing.value = 1
        paths = np.repeat(self.clip_paths, self.n_reps)
        np.random.shuffle(paths)

        for path in paths:
            if self.kill_flag.value:
                break
            s = Sound(path)
            dur = s.getDuration()
            s.play()
            wait(dur)
            wait(self.dur)
        self.playing.value = 0

    def end(self):
        self.kill_flag.value = 1
        #while self.playing.value:
        #    time.sleep(0.025)

if __name__ == '__main__':
    vi = SqueezeInstructions()
    vi.play()
    #vi.end()

    bl = BaselineEyes()
    bl.play()
##

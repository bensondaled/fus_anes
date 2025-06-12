##
import os
import threading
import numpy as np

import fus_anes.config as config

from psychopy.sound import Sound
from psychopy.core import wait
#import psychtoolbox as ptb

# TODO: incorporate saver

class VerbalInstructions():
    def __init__(self):
        self.name = config.name.lower()
        self.audio_path = os.path.join(config.verbal_instructions_path, self.name)
        self.interval = config.verbal_instruction_interval

        self.kill_flag = False

    def get_clip(self, i=None):
        suffix = f'_{i:0.0f}' if i is not None else ''
        filename = f'{self.name}{suffix}.mp3'
        path = os.path.join(self.audio_path, filename)
        if not os.path.exists(path):
            return None
        return path

    def play(self):
        threading.Thread(target=self._play, daemon=True).start()

    def _play(self, with_nums=True):
        idx = 1
        while not self.kill_flag:
            if with_nums:
                clip = self.get_clip(idx)
            else:
                clip = self.get_clip(None)

            s = Sound(clip)
            dur = s.getDuration()
            
            isi = self.interval[0] + np.random.normal(*self.interval[1])

            print(clip, dur, isi)
            s.play() # To do: consider scheduling this play so its timing is more accurate
            wait(dur)

            wait(isi)
            idx += 1

    def end(self):
        self.kill_flag = True

if __name__ == '__main__':
    vi = VerbalInstructions()
    vi.play()

    #vi.end()
##

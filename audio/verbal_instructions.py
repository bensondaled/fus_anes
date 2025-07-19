##
import os
import threading
import time
import numpy as np
import multiprocessing as mp

import fus_anes.config as config
from fus_anes.util import save
from .audio_util import play_tone_precisely, load_audio

import sounddevice as sd
import soundfile as sf

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
    def __init__(self, with_nums=False, saver_buffer=None):
        super(SqueezeInstructions, self).__init__()
        self.name = config.name.lower()
        self.audio_path = config.squeeze_path
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
        save('squeeze', dict(event='play',onset_ts=np.nan, isi=np.nan, dur_words=np.nan, dur_delay=np.nan), self.saver_buffer)
        self.start()

    def run(self):
        idx = 1
        
        fs = config.audio_playback_fs
        beep_t = np.arange(0, int(round(fs * config.squeeze_beep_dur))) / fs
        beep = np.sin(2 * np.pi * config.squeeze_beep_f * beep_t)
        envelope = np.ones_like(beep)
        ramp_samples = int(0.005 * fs)
        ramp = np.linspace(0, 1, ramp_samples)
        envelope[:ramp_samples] *= ramp
        envelope[-ramp_samples:] *= ramp[::-1]
        beep *= envelope
        
        numberless_clip = self.get_clip(None)
        numberless_data, numberless_fs = load_audio(numberless_clip)

        while not self.kill_flag.value:
            if self.with_nums:
                clip = self.get_clip(idx)
                if clip is None:
                    idx = 1
                    continue
                data, fs = load_audio(clip)
            else:
                clip = numberless_clip
                data, fs = numberless_data, numberless_fs

            if config.use_squeeze_beep:
                # add in the beep
                beep_delay_ms = float(np.random.randint(*config.squeeze_beep_delay))
                beep_delay = beep_delay_ms / 1000.0
                delay = np.zeros(int(round(fs * beep_delay)))
                dur_words = len(data) / fs
                dur_delay = len(delay) / fs
                data = np.concatenate([data, delay, beep], axis=0)
            else:
                dur_words = len(data) / fs
                dur_delay = -1.0

            playtime = play_tone_precisely(data, fs)
            isi_ms = np.random.randint(self.interval[0]*1000, self.interval[1]*1000) # NOTE this ISI is from END of instruction unlike other auditory tasks
            save('squeeze', dict(event=os.path.split(clip)[-1], onset_ts=playtime, isi=float(isi_ms), dur_words=dur_words, dur_delay=dur_delay), self.saver_buffer)
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
            data, fs = load_audio(path)
            
            save('bl_eyes', dict(event=os.path.split(path)[-1]), self.saver_buffer)
            sd.play(data, fs)
            sd.wait()
            for _ in range(int(self.dur)):
                time.sleep(1)
                if self.kill_flag.value:
                    break
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

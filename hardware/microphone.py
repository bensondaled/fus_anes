import numpy as np
import os
import multiprocessing as mp
import queue
import threading
import logging
import sys
import h5py
import ctypes
import time
import sounddevice as sd
import soundfile as sf

import fus_anes.config as config
from fus_anes.util import now

sd.default.device = config.audio_in_ch_out_ch

def probe_mics():
    print(sd.query_devices())

if config.THREADS_ONLY:
    mproc = threading.Thread
else:
    mproc = mp.Process

class Microphone(mproc):
    def __init__(self, name, error_queue=None):
        super(Microphone, self).__init__()
        self.save_path = os.path.join(config.data_path, f'{name}_microphone.h5')
        self.error_queue = error_queue or queue.Queue()

        self.n_time_fields = len(now())
        self.save_audio_buffer_shape = config.audio_save_chunk
        self.save_audio_buffer_ts_shape = [config.audio_save_chunk, self.n_time_fields]
        self.hdf_resize_audio = config.audio_hdf_resize
        self.hdf_resize_audio_ts = [config.audio_hdf_resize, self.n_time_fields]
        
        self.kill_flag = mp.Value('b', 0)
        self._on = mp.Value('b', 0)
        self._saving = mp.Value('b', 0)

        self.current_audio_q = mp.Queue()
        self.current_audio = np.zeros(int(config.n_audio_display * 44100), dtype=np.float32)
        self.current_audio_mp = mp.Array(ctypes.c_float, self.current_audio)

        self.n_frames_captured_a = mp.Value('i', 0)
        self.n_frames_queued_a = mp.Value('i', 0)
        self.n_frames_saved_a = mp.Value('i', 0)
        
        self.audio_buffer = mp.Queue()
        
        threading.Thread(target=self.keep_current_audio, daemon=True).start()
        self.start()

    def setup_audio(self):
        self.audio_stream = sd.InputStream(channels=1, device=config.audio_in_ch_out_ch[0], samplerate=44100, blocksize=config.audio_stream_chunk, callback=self.audio_callback)
        self.audio_stream.start()

    def audio_callback(self, data, n, tinfo, flags):
        if self.kill_flag.value:
            return None

        ts = now()
        dat = np.frombuffer(data, dtype=np.float32)
        if self._saving.value:
            self.audio_buffer.put([dat, ts])
            self.current_audio_q.put(dat)
            self.n_frames_captured_a.value += len(dat)
        
    def run(self):
        try:
            self.setup_audio()

            empty_a = np.zeros(self.hdf_resize_audio, dtype=np.float32)
            empty_a_t = np.zeros(self.hdf_resize_audio_ts, dtype=np.float64)

            with h5py.File(self.save_path, 'a') as hfile:
                if 'audio' not in hfile:
                    ds_a = hfile.create_dataset('audio', data=empty_a,
                                                compression='lzf', dtype=np.float32,
                                                maxshape=(None,))
                else:
                    ds_a = hfile['audio']
                
                if 'audio_time' not in hfile:
                    ds_a_t = hfile.create_dataset('audio_time', data=empty_a_t,
                                                compression='lzf', dtype=np.float64,
                                                maxshape=(None,None))
                else:
                    ds_a_t = hfile['audio_time']

            self.hdf_idx_a = 0
            n_dumped_audio = 0

            self.save_audio_buffer = np.zeros(self.save_audio_buffer_shape, dtype=np.float32)
            self.save_audio_buffer_t = np.zeros(self.save_audio_buffer_ts_shape, dtype=np.float64)

            self._saving.value = 1
            self._on.value = 1
            finished = False
            
            warned_mic = 0

            while True:
                try:
                    aud, ts = self.audio_buffer.get(block=False)
                    if len(aud) < config.audio_stream_chunk:
                        aud = np.pad(aud, (0, config.audio_stream_chunk - len(aud)), 'constant', constant_values=(0, 0)).astype(np.float32)
                    self.save_audio_buffer[n_dumped_audio : n_dumped_audio + config.audio_stream_chunk] = aud
                    self.save_audio_buffer_t[n_dumped_audio : n_dumped_audio + config.audio_stream_chunk, :] = ts
                    n_dumped_audio += config.audio_stream_chunk
                    self.n_frames_queued_a.value += len(aud)

                    if self.n_frames_captured_a.value - self.n_frames_saved_a.value > self.save_audio_buffer_shape * 3:
                        if warned_mic == 3:
                            logging.warning('Warnings from mic continue, will spare the ongoing alerts.')
                            warned_mic += 1
                        elif warned_mic > 3:
                            pass
                        else:
                            logging.warning(f'Falling behind on mic saving:\nCaptrd\t{self.n_frames_captured_a.value}\nQueued\t{self.n_frames_queued_a.value}\nSaved\t{self.n_frames_saved_a.value}\nDelta\t{self.n_frames_captured_a.value-self.n_frames_saved_a.value}\n')
                            warned_mic += 1
                    if n_dumped_audio == config.audio_save_chunk:
                        self.empty_save_buffer()
                        n_dumped_audio = 0
                        
                except queue.Empty:
                    if self.kill_flag.value:
                        if not finished:
                            self.empty_save_buffer(N=n_dumped_audio, final=True)
                            finished = True
                
                if finished:
                    self.audio_stream.stop()
                    self.audio_stream.close()
                    self._on.value = False
                    break

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.error_queue.put(f'Mic main (line {exc_tb.tb_lineno}): {str(e)}')
            
    def empty_save_buffer(self, N=None, final=False):

        dat, dat_t = self.save_audio_buffer, self.save_audio_buffer_t

        if N is not None:
            dat = dat[:N]
            dat_t = dat_t[:N, :]

        with h5py.File(self.save_path, 'a') as hfile:
            ds = hfile['audio']
            ds_t = hfile['audio_time']
            hdf_idx = self.hdf_idx_a
            hdfrs = self.hdf_resize_audio
            
            if hdf_idx + len(dat) > ds.shape[0]:
                ds.resize(ds.shape[0] + hdfrs, axis=0)
                ds_t.resize(ds_t.shape[0] + hdfrs, axis=0)
            ds[hdf_idx:hdf_idx+len(dat)] = dat
            ds_t[hdf_idx:hdf_idx+len(dat), :] = dat_t
            
            self.hdf_idx_a += len(dat)
            self.n_frames_saved_a.value = hdf_idx
        
            # cut off extra
            if final:
                hdf_idx = self.hdf_idx_a
                ds.resize(hdf_idx, axis=0)
                ds_t.resize(hdf_idx, axis=0)
    
    def keep_current_audio(self):  
        while not self.kill_flag.value:    
            try:
                aud = self.current_audio_q.get(block=True)                         
                self.current_audio = np.roll(self.current_audio, -len(aud))
                self.current_audio[-len(aud):] = aud
                self.current_audio_mp[:] = self.current_audio.copy()
            except queue.Empty:
                pass
               
    def get_current_audio(self):            
        return np.frombuffer(self.current_audio_mp.get_obj(), dtype=np.float32)

    def end(self):
        self.kill_flag.value = 1
        while self._on.value:
            time.sleep(0.250)

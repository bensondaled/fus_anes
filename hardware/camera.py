import numpy as np
import os
import multiprocessing as mp
import queue
import threading
import h5py
import time

import fus_anes.config as config
from fus_anes.util import now, now2

if config.THREADS_ONLY:
    mproc = threading.Thread
else:
    mproc = mp.Process

class Camera(mproc):
    def __init__(self, name):
        super(Camera, self).__init__()
        self.save_path = os.path.join(config.data_path, f'{name}_camera.avi')

        self.kill_flag = mp.Value('b', 0)
        self._on = mp.Value('b', 0)
        self.current_frame_q = mp.Queue()
        self.current_frame = None

        threading.Thread(target=self.keep_current, daemon=True).start()
        self.start()

    def run(self):
        vc = cv2.VideoCapture(0)
        vw_fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vw = cv2.VideoWriter(self.save_path, vw_fourcc, 20.0, (640, 480))

        self._on.value = 1
        finished = False

        while not self.kill_flag.value:
            _, frame = vc.read()
            vw.write(frame)
            self.current_frame_q.put(frame)
        
        vc.release()
        vw.release()
        self._on = False

    def keep_current(self):
        while not self.kill_flag.value:
            try:
                self.current_frame = self.current_frame_q.get(block=True)
            except queue.Empty:
                pass

    def end(self):
        self.kill_flag.value = 1
        while self._on.value:
            time.sleep(0.010)

##
import cv2
import numpy as np
import threading
import multiprocessing as mp
import time
import os
import sys
import queue

np.set_printoptions(suppress=True, threshold=sys.maxsize, precision=15) # for saving timestamp text

from fus_anes.util import now
import fus_anes.config as config

def a2s(a):
    return np.array2string(a,
                           precision=15,
                           separator=',',
                           floatmode='maxprec',
                           formatter={'float_kind': lambda x: f'{x:.15f}'},
                           )

def list_cameras():
    index = 0
    available_cameras = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        else:
            available_cameras.append(index)
        cap.release()
        index += 1
    return available_cameras

if config.THREADS_ONLY:
    mproc = threading.Thread
else:
    mproc = mp.Process
    
class Camera(mproc):
    def __init__(self, name, error_queue=None,):
        super(Camera, self).__init__()
        self.name = name
        self.save_path = os.path.join(config.data_path, f'{name}_camera')
        os.makedirs(self.save_path, exist_ok=True)
        self.save_path_ts = os.path.join(self.save_path, f'{name}_camera_timestamps.txt')
        
        self.frame_buffer = mp.Queue()
        self.kill_flag = mp.Value('b', 0)
        self._on = mp.Value('b', 0)
        self.n_frames_captured = mp.Value('i', 0)
        self.n_frames_saved = mp.Value('i', 0)
        
        self.current_frame_q = mp.Queue()
        self.current_frame = None
        
        self.error_queue = error_queue
        
        mproc(target=self.stream_video, daemon=True).start()
        threading.Thread(target=self.keep_current, daemon=True).start()
        self.start()

    def run(self):
        def new_vw(idx=0):
            if isinstance(config.fourcc, str) and len(config.fourcc)==4:
                fourcc = cv2.VideoWriter_fourcc(*config.fourcc)
            else:
                fourcc = -1

            path = os.path.join(self.save_path, f'{self.name}_camera_{idx}.{config.mov_ext}')
            vw = cv2.VideoWriter(path, 
                fourcc,
                30, # frame rate, relatively irrelevant bc will use timestamps
                (int(config.cam_frame_size[1]*config.cam_resize), int(config.cam_frame_size[0]*config.cam_resize)),
                isColor=True)
            if not vw.isOpened():
                self.error_queue.put('Camera videowriter didnt open')
            return vw
    
        try:
                      
            finished = False # video, audio
            self._on.value = 1
            
            n_ts_fields = len(now())
            ts_buffer = np.zeros([1000, n_ts_fields], dtype=float)
            ts_buffer_idx = 0
            
            start_time = None
            vw_idx = 0
            vw = new_vw(vw_idx)
            
            while True:
                # video
                try:
                    frame, ts = self.frame_buffer.get(block=False)
                    vw.write(frame)
                    ts_buffer[ts_buffer_idx, :] = ts
                    ts_buffer_idx += 1
                    self.n_frames_saved.value += 1
                    
                    if start_time is None:
                        start_time = ts[0]
                    
                    if ts_buffer_idx == len(ts_buffer):
                        with open(self.save_path_ts, 'a') as f:
                            f.write(a2s(ts_buffer))
                        ts_buffer_idx = 0
                        
                    if ts[0] - start_time > config.cam_file_duration:
                        vw.release()
                        vw_idx += 1
                        start_time = ts[0]
                        vw = new_vw(vw_idx)
                    
                    #print(self.n_frames_captured.value, self.n_frames_saved.value)
                        
                except queue.Empty:
                    if self.kill_flag.value:
                        finished = True
                
                
                if finished:
                    with open(self.save_path_ts, 'a') as f:
                        f.write(a2s(ts_buffer[:ts_buffer_idx]))
                    vw.release()
                    self._on.value = False
                    break
        except Exception as e:
            self.error_queue.put(f'Camera main: {str(e)}')
            
    def stream_video(self):
        if config.cam_resize not in [None, 1]:
            resize = config.cam_resize
        else:
            resize = False

        try:
            vc = cv2.VideoCapture(0)

            frame_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frame_width != config.cam_frame_size[1] or frame_height != config.cam_frame_size[0]:
                self.error_queue.put(f'Frame dims mismatch: {frame_width}, {frame_height}')
                return

            while self.kill_flag.value == 0:
                ts = now()
                success, frame = vc.read()
                    
                if resize:
                    frame = cv2.resize(frame, None, fx=resize, fy=resize)
                
                if success:
                    self.frame_buffer.put([frame, ts])
                    self.current_frame_q.put(frame)
                    self.n_frames_captured.value += 1
            vc.release()
        except Exception as e:
            self.error_queue.put(f'Camera stream: {str(e)}')
                

    def keep_current(self):
        while not self.kill_flag.value:
            try:
                frame = self.current_frame_q.get(block=True)
                self.current_frame = frame
            except queue.Empty:
                pass

    def end(self):
        self.kill_flag.value = 1
        while self._on.value:
            time.sleep(0.100)
            
if __name__ == '__main__':
    cam = Camera('mytest', error_queue=mp.Queue())
    while cam.n_frames_captured.value == 0:
        time.sleep(0.050)
    for i in range(60):
        time.sleep(1)
        print(i)
    cam.end()

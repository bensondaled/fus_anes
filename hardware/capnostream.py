import serial
import threading
import numpy as np
import time
import multiprocessing as mp
import queue

from fus_anes.util import now, now2
import fus_anes.config as config

if config.THREADS_ONLY:
    mproc = threading.Thread
else:
    mproc = mp.Process

class DummyPort():
    def __init__(self):
        pass
    def write(self, *args):
        pass
    def read(self, *args):
        return b''
    def close(self):
        pass

class Capnostream(mp.Process):
    '''
    Commands:
        - Enable communication protocol: code 1, length 1, no data
        - Disable communication protocol: code 2, length 1, no data
        - Inquire numeric: code 3, length 2, data etco2=1, fico2=2, rr=3, spo2=4, pulse=5
        - Start realtime communication: code 4, length 1, no data
        - Stop realtime communication: code 5, length 1, no data

    Realtime data:
        - CO2 wave message:
            - code 0
            - 4 data bytes: wave_message_number, co2 value (2 bytes, integer and frac /256), fast status (bitwise)
        - Numeric message:
            - code 1
            - 27 data bytes:
                 timestamp, 4, #1-4
                 etco2, 1, #5
                 fico2, 1, #6
                 rr, 1, #7
                 spo2, 1, #8
                 pulse, 1, #9
                 slow status, 1, #10
                 events index, 3, #11-13
                 co2 alarms, 1, #14
                 spo2 alarms, 1, #15
                 no breath, 1, #16
                 etco2 high, 1, #17
                 etco2 low, 1, #18
                 rr high, 1, #19
                 rr low, 1, #20
                 fico2 high, 1, #21
                 spo2 high, 1, #22
                 spo2 low, 1, #23
                 pulse high, 1, #24
                 pulse low, 1, #25
                 co2 units, 1, #26
                 co2 status, 1, #27
    '''

    def __init__(self, port=config.capnostream_port, saver_obj_buffer=None, error_queue=None,):
        super(Capnostream, self).__init__()
        
        self.saver_obj_buffer = saver_obj_buffer
        self.port = port
        self.kill_flag = mp.Value('b', 0)
        self._on = mp.Value('b', 0)

        self.save_buffer_dims = [config.capnostream_save_buffer_length, 5]
        self.co2_read_buffer = mp.Queue()
        
        self.live_buffer = np.zeros(config.capnostream_live_buffer_length)
        self.live_buffer_mp = mp.Array('d', self.live_buffer.copy().ravel())
        
        self.error_queue = error_queue

        self.start()

    def initialize_device(self):
        if config.TESTING:
            self.port = DummyPort()
        else:
            self.port = serial.Serial(self.port, baudrate=9600,
                            bytesize=serial.EIGHTBITS,
                            stopbits=serial.STOPBITS_ONE,
                            parity=serial.PARITY_NONE,
                            timeout=0.2,)
                            
        msg_stop_protocol = bytes([0x85, 1, 5, 1^5])
        msg_stop_realtime = bytes([0x85, 1, 2, 1^2])
        self.port.write(msg_stop_protocol)
        self.port.write(msg_stop_realtime)
        time.sleep(0.1)

        msg_start_protocol = bytes([0x85, 1, 1, 1^1])
        self.port.write(msg_start_protocol)
        resp = self.port.read(34) # device id etc message
        if len(resp) != 34:
            raise Exception('Failed to initialize CO2 module. Not running.')
        assert resp[0]==0x85
        
        msg_start_realtime = bytes([0x85, 1, 4, 1^4])
        self.port.write(msg_start_realtime)

    def run(self):
        try:
            self.initialize_device()
                
            threading.Thread(target=self.acquisition_loop, daemon=True).start()

            buf_idx = 0
            self.save_buffer = np.zeros(self.save_buffer_dims).astype(float)
            while True:
                
                try:
                    msg, ts, cs = self.co2_read_buffer.get(block=False)
                    
                    # update save buffer
                    self.save_buffer[buf_idx, :3] = msg
                    self.save_buffer[buf_idx, -2:] = [ts, cs]
                    buf_idx += 1
                    
                    # update live buffer
                    self.live_buffer = np.roll(self.live_buffer, -1)
                    self.live_buffer[-1] = msg[1] # co2 value
                    self.live_buffer_mp[:] = self.live_buffer.copy()
                    
                    if buf_idx == len(self.save_buffer):
                        self.empty_save_buffer()
                        buf_idx = 0
                        
                except queue.Empty:
                    if self.kill_flag.value:
                        self.empty_save_buffer(N=buf_idx)
                        break
                        
            msg_stop_protocol = bytes([0x85, 1, 5, 1^5])
            msg_stop_realtime = bytes([0x85, 1, 2, 1^2])
            self.port.write(msg_stop_protocol)
            self.port.write(msg_stop_realtime)
            self.port.close()
            self._on.value = False
        except Exception as e:
            self.error_queue.put(f'Capnostream: {str(e)}')

    def acquisition_loop(self):
        while self.kill_flag.value == 0:
            try:
                code, msg, ts, cs = self.read()
            except:
                continue

            if code != 0: # was not a co2 value read:
                continue

            wv_msg_num, co2_0, co2_1, fast = msg
            co2 = co2_0 + co2_1 / 256
            msg = [wv_msg_num, co2, fast]

            self.co2_read_buffer.put([msg, ts, cs])

    def read(self):
        while True:
            byt = self.port.read(1)
            if byt == b'\x85':
                break
        ts, cs = now(), now2()
        bodysize = int.from_bytes(self.port.read(1))
        body = self.port.read(bodysize)
        checksum = self.port.read(1)
        if len(body) != bodysize:
            print('Did not get expected length response, expected {bodysize}, got {len(body)}, skipping sample')
            return None
        code, msg = body[0], body[1:]
        if code == 0 and bodysize != 5:
            print('Did not get expected length response, expected 5 for co2 message of code 0, got {bodysize}, skipping sample')
            return None
        msg = [m for m in msg]
        return code, msg, ts, cs
    
    def empty_save_buffer(self, N=0):
        if self.saver_obj_buffer is None:
            return
        self.saver_obj_buffer.put(['co2',
                                   self.save_buffer[-N:, :-2].copy(),
                                   self.save_buffer[-N:, -2].copy(),
                                   self.save_buffer[-N:, -1].copy(),
                                   ['wave_msg_num', 'co2_value', 'fast_status']],
                                  )
        self.n_new = 0
        
    def get_current(self):            
        return np.frombuffer(self.live_buffer_mp.get_obj())
                    
    def end(self):
        self.kill_flag.value = 1
        while self._on.value:
            time.sleep(0.010)



import time
import sys
import pylsl
time_time_module = time
if sys.platform.startswith('win'):
    import win_precise_time as wpt
    time_time_module = wpt


def now(minimal=False):
    lsl_stamp = pylsl.local_clock()
    time_stamp = time_time_module.time()
    pc_stamp = time.perf_counter()
    if minimal:
        return lsl_stamp
    else:
        return [lsl_stamp, time_stamp, pc_stamp]


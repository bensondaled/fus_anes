import time
import sys

if sys.platform.startswith('win'):
    import win_precise_time as wpt

def now():
    if sys.platform.startswith('win'):
        return wpt.time()
    else:
        return time.time()
def now2():
    return time.perf_counter()


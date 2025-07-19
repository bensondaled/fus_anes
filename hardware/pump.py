'''
https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers?tab=downloads
https://www.harvardapparatus.com/media/harvard/pdf/552222_Pump_22_Manual.pdf
https://www.prolific.com.tw/US/ShowProduct.aspx?p_id=225&pcid=41  - not used other than to check its status. did not install

final solution for getting rs22 usb adapter working: uninstall it all then replug without any installations
'''

import serial
import warnings

import fus_anes.config as config

class DummyPort():
    def __init__(self):
        pass
    def write(self, *args):
        if config.TESTING:
            print(f'Writing {args}')
            return
        pass
    def read(self, *args):
        return b''
    def readline(self, *args):
        return b''
    def reset_output_buffer(self):
        pass
    def reset_input_buffer(self):
        pass
    def close(self):
        pass

class Pump():
    def __init__(self, port=config.pump_port, error_queue=None, saver=None):
        self.saver = saver
        self.error_queue = error_queue
        if config.TESTING:
            self.port = DummyPort()
        else:
            self.port = serial.Serial(port, baudrate=9600,
                            bytesize=serial.EIGHTBITS,
                            stopbits=serial.STOPBITS_TWO,
                            parity=serial.PARITY_NONE,
                            timeout=1.0,)
                        
        self.set_diameter(config.syringe_diam)

        self.current_infusion_rate = 0
    
    def infuse(self, rate):
        # rate given in ml/min
        if rate < config.pump_rate_min:
            self.command('stp')
        else:
            self.set_flow_rate(rate, 'mL/min')
            self.command('run')
        self.current_infusion_rate = rate
    
    def write(self, msg):
        if self.saver is not None:
            self.saver.write('pump', dict(msg=f'{msg:<20}'))

        self.port.reset_output_buffer()
        self.port.reset_input_buffer()
        self.port.write(f'{msg}\r'.encode())
        
    def read(self, *args):
        if config.TESTING:
            print(f'Reading port')
            return

        return self.port.read(*args).decode('utf-8')
    
    def get_value(self, key):
        '''
        DIA : diameter
        RAT : flow rate
        VOL : volumne accum
        VER : version
        TAR : target volume
        STATUS : status
        '''
        if key == 'STATUS':
            key = ''

        self.write(key)
        
        s = self.read(8)
        self.read_reply()
        return float(s)
    
    def command(self, cmd):
        ''' 
        RUN : start
        STP : stop
        CLV : clear volume accum
        CLT : clear target
        REV : reverse
        '''
        self.write(cmd)
        self.read_reply()
    
    def read_reply(self):
        try:
            if config.TESTING:
                print(f'Reading reply')
                return

            keys = {
                    ':': 'stopped',
                    '>': 'running',
                    '<': 'reverse',
                    '*': 'stalled' }
            s = ''
            for i in range(5): # Try reading up to 5 characters for response
                s = self.read(1)
                if s in keys:
                    return keys[s]
                elif s == 'O' or s == '?':
                    break

            s += self.port.readline().decode('utf-8')
            if s.startswith('OOR'):
                warnings.warn('Pump: Out of range')
                self.error_queue.put(f'Pump: out of range')
            elif s.startswith('?'):
                warnings.warn('Pump: Unrecognized command')
                self.error_queue.put(f'Pump: command unrecognized')
            else:
                warnings.warn(f'Pump: Unrecognized response: {s}')
                self.error_queue.put(f'Pump: response unrecognized')
        except Exception as e:
            self.error_queue.put(f'Pump: {str(e)}')        
    
    def get_unit(self):
        units = {
                'ML/H': 'mL/hr',
                'ML/M': 'mL/min',
                'UL/H': 'uL/hr',
                'UL/M': 'uL/min' }
        self.write('RNG')
        print(self.read(2)) # Read CR/LF
        unit = self.read(2)
        if '?' in unit: return None
        unit += self.read(2)
        self.read(2) # Read CR/LF
        self.read_reply()
        return units[unit]

    def set_diameter(self, value):
        self.write(f'MMD {value}') # mm
        return self.read_reply()

    def set_flow_rate(self, value, unit):
        units = {
                'uL/min': 'ULM',
                'mL/min': 'MLM',
                'uL/hr' : 'ULH',
                'mL/hr' : 'MLH' }
        
        self.write(f'{units[unit]} {value:0.3f}')
        self.read_reply()

    def set_direction(self, direction):
        status = self.get_value('status')
        if (status == 'running' and direction == -1) or \
           (status == 'reverse' and direction == 1) or \
            status == 'stalled':
               self.command('REV')

    def set_target_volume(self, value):
        self.write(f'MLT {value}')
        self.read_reply()
        
    def end(self):
        self.command('stp')
        self.port.close()


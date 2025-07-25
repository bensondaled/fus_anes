import threading
import numpy as np
import time
import warnings
import re
import os
from PyQt5 import QtWidgets as qtw
from PyQt5.QtCore import QTimer
from functools import wraps

from fus_anes.util import nanpow2db, now
import fus_anes.config as config
from fus_anes.interface import Interface
from .session import Session

def str2num(s):
    try:
        s = float(s)
    except:
        s = 0
    return s

def require_session(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.running:
            return lambda *a, **kw: None
        if self.session and self.session.running:
            return func(self, *args, **kwargs)
    return wrapper

class Controller():
    def __init__(self, app):
        self.ui = Interface(app=app)

        # bindings
        self.ui.closing.connect(self.end)
        self.ui.b_sesh.clicked.connect(self.session_toggle)
        self.ui.b_run_baseline.clicked.connect(self.toggle_baseline)
        self.ui.b_run_squeeze.clicked.connect(self.toggle_squeeze)
        self.ui.b_run_oddball.clicked.connect(self.toggle_oddball)
        self.ui.b_run_chirp.clicked.connect(self.toggle_chirp)
        self.ui.b_run_ssep.clicked.connect(self.run_ssep)
        self.ui.b_run_steady.clicked.connect(self.run_steady)
        self.ui.b_run_vigilance.clicked.connect(self.run_vigilance)
        self.ui.b_bolus.clicked.connect(self.bolus)
        self.ui.b_infusion.clicked.connect(self.infuse)
        self.ui.b_project.clicked.connect(self.project)
        self.ui.b_project.clicked.connect(self.project)
        self.ui.b_simulate.clicked.connect(self.simulate)
        self.ui.b_set_tci_target.clicked.connect(self.set_tci_target)
        self.ui.b_clear_tci_queue.clicked.connect(self.clear_tci_queue)
        self.ui.b_reset_xlim.clicked.connect(self.reset_xlim)
        self.ui.b_toggle_raw.clicked.connect(self.toggle_raw)
        self.ui.b_toggle_video.clicked.connect(self.toggle_video)
        self.ui.b_marker.clicked.connect(self.mark)
        self.ui.b_lor.clicked.connect(self.lor)
        self.ui.b_ror.clicked.connect(self.ror)
        self.ui.t_lopass.editingFinished.connect(self.update_filters)
        self.ui.t_hipass.editingFinished.connect(self.update_filters)
        self.ui.t_notch.editingFinished.connect(self.update_filters)
        self.ui.timeline_nav_lr.sigRegionChangeFinished.connect(self.drag_timeline_nav)
        
        for chan_name, obj_list in self.ui.spect_time_selects.items():
            for idx, v_bar in enumerate(obj_list):
                v_bar.sigRegionChangeFinished.connect(lambda *args, name=chan_name, idx=idx: self.select_spect_time(*args, selection_idx=idx, figname=name))
        for chan_name, obj_list in self.ui.spect_freq_selects.items():
            for idx, h_bar in enumerate(obj_list):
                h_bar.sigRegionChangeFinished.connect(lambda *args, name=chan_name, idx=idx: self.update_spect_freq_selects(sel_idx=idx, chan_name=name, updated='regions'))
        for chan_name, obj_list in self.ui.spect_freq_select_valboxes.items():
            for idx, (lim0, lim1) in enumerate(obj_list):
                lim0.editingFinished.connect(lambda *_, chan_name=chan_name, idx=idx: self.update_spect_freq_selects(chan_name, updated='txts', sel_idx=idx))
                lim1.editingFinished.connect(lambda *_, chan_name=chan_name, idx=idx: self.update_spect_freq_selects(chan_name, updated='txts', sel_idx=idx))

        ''' generic code for clicking a plot at arbitrary point
        obj.scene().sigMouseClicked.connect(lambda *args, name=name: self.click_spect(*args, figname=name))
        obj = self.ui.plot_objs[figname]
        vb = obj.plotItem.vb
        scene_coords = event.scenePos()
        mouse_point = vb.mapSceneToView(scene_coords)
        x, y = mouse_point.x(), mouse_point.y()
        '''

        
        # run
        self.session = None
        self.running = True
        self.raw_frame_state = 'split'
        self.last_spect_memory = None
        self.eeg_raw_dat = np.zeros([int(round(config.raw_eeg_display_dur * config.fs)), config.n_channels+config.eeg_n_timefields]) * np.nan
        self.eeg_raw_idx = 0
        self.eeg_raw_xvals = np.arange(self.eeg_raw_dat.shape[0]) / config.fs
        self.next_tci_inst = None
        
        # update functions
        self.timers = {}
        self.repeat(self.update_eeg_raw, 15)
        self.repeat(self.update_eeg_spect,
                    1000 * int(config.spect_update_interval / config.fs))
        self.repeat(self.update_generic, 250) 
        self.repeat(self.update_tci_inst, 4000) 
        self.repeat(self.update_tci, 1500)
        self.repeat(self.update_timeline, 250)
        self.repeat(self.update_video, 30)
        self.repeat(self.update_errors, 1000)

    def repeat(self, function, interval):
        timer = QTimer()
        timer.timeout.connect(function)
        timer.start(interval)
        self.timers[function.__name__] = timer
    
    @require_session
    def drag_timeline_nav(self, *args):
        x0, x1 = self.ui.timeline_nav_lr.getRegion()
        self.ui.user_zoompan(xlim=[x0, x1])
    
    @require_session
    def toggle_baseline(self, *args):
        self.session.toggle_baseline()
    
    @require_session
    def select_spect_time(self, obj, selection_idx=0, figname=''):
        x0, x1 = obj.getRegion()

        disp_idx = int(re.match(r'eeg_spect_(.*)', figname).group(1))
        if self.last_spect_memory is None:
            data = self.session.eeg.get_spect_memory() # already in dB
        else:
            data = self.last_spect_memory
        chan_idx = self.ui.get_chan_selections(disp_idx)
        dat = data[chan_idx]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'Mean of empty slice')
            dat = np.nanmean(dat, axis=0) # avg over selected channels

        sp_Ts = self.session.eeg.spect_interval / self.session.eeg.fs
        t0 = self.session.eeg.first_spect_time.value
        _, freqs = self.session.eeg.spect_memory_tf

        xvals = np.arange(0, dat.shape[-1]) * sp_Ts
        xvals -= self.session.running - t0

        ci_0 = np.argmin(np.abs(x0-xvals))
        ci_1 = np.argmin(np.abs(x1-xvals))
        slc = dat[:, ci_0:ci_1+1]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'Mean of empty slice')
            res = np.nanmean(slc, axis=1)
        if (x0<0 and x1<0) or (np.all(np.isnan(res))) or np.min(np.abs(x0-xvals))>2*(config.spect_update_interval/config.fs):
            res[:] = 0

        #res -= np.nanmin(res)
        
        self.ui.update_eeg_psd(disp_idx,
                                   selection_idx,
                                   data=res,
                                   xvals=freqs,
                                  )
    
    @require_session
    def mark(self, event):
        t = now(minimal=True)
        text = self.ui.t_marker.text()
        self.session.add_marker([t, text])
    
    @require_session
    def lor(self, event):
        t = now(minimal=True)
        text = 'lor'
        self.session.add_marker([t, text])
    
    @require_session
    def ror(self, event):
        t = now(minimal=True)
        text = 'ror'
        self.session.add_marker([t, text])
    
    @require_session
    def bolus(self, event):
        t = now(minimal=True)
        dose = self.ui.t_bolus.text()
        try:
            dose = int(dose)
        except:
            dose = 0

        if dose == 0:
            return

        self.session.tci.bolus(dose)

    @require_session
    def clear_tci_queue(self, event):
        self.session.tci.clear_instruction_queue()
    
    @require_session
    def simulate(self, event):
        b = str2num(self.ui.t_bolus.text())
        i = str2num(self.ui.t_infusion.text())

        sim_time, sim_vals = self.session.tci.simulation(infusion=i, bolus=b)
        sim_time += now(minimal=True)
        sim_time -= self.session.running
        self.ui.update_tci_data(sim_time, sim_vals, kind='sim')
    
    @require_session
    def project(self, event):
        proj_time, proj_vals = self.session.tci.get_projection()
        proj_time += now(minimal=True)
        proj_time -= self.session.running
        self.ui.update_tci_data(proj_time, proj_vals, kind='proj')

    @require_session
    def infuse(self, event):
        t = now(minimal=True)
        dose = str2num(self.ui.t_infusion.text())
        self.session.tci.infuse(dose)

    @require_session
    def set_tci_target(self, event):
        target = str2num(self.ui.t_set_tci_target.text())
        self.session.tci.goto(target)

    @require_session
    def update_timeline(self):
        dt = now(minimal=True) - self.session.running
        self.ui.update_timeline(dt) # moving ticker
        
        #boluses = [t-self.session.running for t, d in self.session.boluses]
        #infusions = [t-self.session.running for t, d in self.session.infusions]
        #self.ui.update_meds(boluses=boluses, infusions=infusions)
        
        markers = [(t-self.session.running, txt) for t, txt in self.session.markers]
        self.ui.update_markers(markers)
    
    def toggle_raw(self):
        raw_obj = self.ui.raw_panel
        chan_obj = self.ui.chan_panel

        if self.raw_frame_state == 'split':
            raw_obj.hide() # check with isHidden
            chan_obj.show() # check with isHidden
            self.raw_frame_state = 'chan'
        elif self.raw_frame_state == 'chan':
            raw_obj.show() # check with isHidden
            chan_obj.hide() # check with isHidden
            self.raw_frame_state = 'raw'
        elif self.raw_frame_state == 'raw':
            raw_obj.show() # check with isHidden
            chan_obj.show() # check with isHidden
            self.raw_frame_state = 'split'
            
    def toggle_video(self):
        if self.ui.vm.isVisible():
            self.ui.vm.setVisible(False)
        else:
            self.ui.vm.setVisible(True)
    
    @require_session
    def reset_xlim(self, *args, **kw):
        dt = now(minimal=True) - self.session.running
        
        x0 = max(0, dt - config.timeline_duration + config.timeline_advance)
        x1 = max(x0+config.timeline_duration, dt + config.timeline_advance)

        self.ui.reset_xlim(x0, x1)
    
    @require_session
    def update_tci(self):
        tci_time, tci_vals = self.session.tci.history
        tci_time -= self.session.running
        self.ui.update_tci_data(tci_time, tci_vals, kind='hist')
        
        '''
        sim_idx, sim_vals = self.session.simulation_idx, self.session.simulated_tci_vals
        tci_time = np.array(tci_time) - self.session.running
        sim_time = tci_time[sim_idx] + np.arange(len(sim_vals))

        prot_idx, prot_vals = self.session.prot_sim_idx, self.session.prot_sim_vals
        prot_time = tci_time[prot_idx] + np.arange(len(prot_vals))
        '''
        
        tci_rate, pump_rate = self.session.tci.infusion_rate
        self.ui.l_infusion_rate.setText(f'{tci_rate:0.0f}mcg/kg/min = {pump_rate:0.3f}ml/min')

    @require_session
    def update_eeg_raw(self):
        data, nadd = self.session.eeg.get_memory(with_idx=True)
        if nadd == 0:
            return
        add = data[-nadd:, :] 

        overflow = self.eeg_raw_idx + nadd - self.eeg_raw_dat.shape[0]
        if overflow > 0:
            try:
                fits = max(0, self.eeg_raw_dat.shape[0] - self.eeg_raw_idx)
                self.eeg_raw_dat[self.eeg_raw_idx:] = add[:fits]
                self.eeg_raw_dat[0 : overflow] = add[fits:]
                self.eeg_raw_idx = overflow
            except:
                pass # just rare cases where delay causes too much to be in overflow. easiest to just let it pass and move on
        else:
            self.eeg_raw_dat[self.eeg_raw_idx : self.eeg_raw_idx + nadd] = add
            self.eeg_raw_idx += nadd
        
        self.ui.update_eeg_raw(xdata=self.eeg_raw_xvals,
                               ydata=self.eeg_raw_dat,
                               vline=self.eeg_raw_xvals[self.eeg_raw_idx-1],
                               )
        
    @require_session
    def update_eeg_spect(self):
        data = self.session.eeg.get_spect_memory() # already in dB
        if np.all(np.isnan(data)):
            return

        self.last_spect_memory = data # for use in select_spect_time
        sp_Ts = config.spect_update_interval / config.fs
        t0 = self.session.eeg.first_spect_time.value
        _, freqs = self.session.eeg.spect_memory_tf

        times = np.arange(0, data.shape[-1]) * sp_Ts
        times -= self.session.running - t0

        self.ui.update_eeg_spect(data=data,
                                 xvals=times,
                                 yvals=freqs,
                                )

    def session_toggle(self, event):
        if self.ui.b_sesh.isEnabled() == False:
            return

        if self.session is None:
            self.ui.b_sesh.setText('(Session initializing)')
            self.ui.setEnabled(False)
            
            self.ui.splash(True, f'Launching session...')
            def _start_sesh():
                self.session = Session()
                self.session.run()
            QTimer.singleShot(100,
                lambda: threading.Thread(target=_start_sesh).start())
        
        elif self.session is not None:
            self.ui.b_sesh.setText('(Session ending)')
            self.ui.setEnabled(False)
            self.ui.splash(True, 'Session ending...')
            #self.ui.reset_runtime_vars()
            QTimer.singleShot(100, self.session.end)

    @require_session
    def toggle_squeeze(self, event):
        self.session.toggle_squeeze()
    
    @require_session
    def toggle_oddball(self, event):
        self.session.toggle_oddball()
    
    @require_session
    def toggle_chirp(self, event):
        self.session.toggle_chirp()
    
    @require_session
    def run_ssep(self, event):
        if not self.ui.b_run_ssep.running:
            self.session.add_marker([now(minimal=True), 'ssep stop'])
        else:
            self.session.add_marker([now(minimal=True), 'ssep start'])
    
    @require_session
    def run_steady(self, event):
        if not self.ui.b_run_steady.running:
            self.session.add_marker([now(minimal=True), 'steady stop'])
        else:
            self.session.add_marker([now(minimal=True), 'steady start'])
    
    @require_session
    def run_vigilance(self, event):
        if not self.ui.b_run_vigilance.running:
            self.session.add_marker([now(minimal=True), 'vigilance stop'])
        else:
            self.session.add_marker([now(minimal=True), 'vigilance start'])
    
    @require_session
    def update_filters(self):
        lo = self.ui.t_lopass.text()
        hi = self.ui.t_hipass.text()
        notch = self.ui.t_notch.text()
        try:
            lo = float(lo)
            if lo == 0:
                lo = config.fs
                self.ui.t_hipass.setText(str(lo))
        except:
            lo = config.fs
            self.ui.t_hipass.setText(str(lo))

        try:
            hi = float(hi)
            if hi == 0:
                hi = 1e-6
        except:
            hi = 1e-6
            self.ui.t_hipass.setText('0')
        if hi >= lo:
            hi = lo - 1
            self.ui.t_hipass.setText(str(hi))

        try:
            notch = float(notch)
            if notch<=0:
                notch = config.fs / 2 # an irrelevant freq wrt analyses but within computable range
        except:
            notch = config.fs / 2

        self.session.eeg.set_filters(lo=lo, hi=hi, notch=notch)
    
    def update_generic(self):
        if not self.running:
            return

        if self.session and self.session.running:
            if self.ui.b_sesh.text() == '(Session initializing)':
                self.ui.l_sesh.setText(str(self.session.pretty_name))
                self.ui.b_sesh.setText('End session')
                self.ui.splash(False)
                self.ui.setEnabled(True)
            
            words = ['baseline', 'squeeze', 'oddball', 'chirp']
            objs = [self.session.baseline_eyes, self.session.squeeze, self.session.oddball, self.session.chirp]
            buts = [self.ui.b_run_baseline, self.ui.b_run_squeeze, self.ui.b_run_oddball, self.ui.b_run_chirp]
            for word, obj, but in zip(words, objs, buts):
                if obj is None and but.running:
                    but.toggle()
        
            if self.next_tci_inst is not None:
                rate, ts = self.next_tci_inst
                time_until = max(0, ts - now(minimal=True))
                inst_str = f'Next rate: {rate:0.0f} in {time_until:0.0f}s'
                self.ui.l_tci_inst.setText(inst_str)

        if self.ui.b_sesh.text() == '(Session ending)' and self.session.completed:
            self.session = None
            self.ui.l_sesh.setText('(none)')
            self.ui.b_sesh.setText('New session')
            self.ui.splash(False)
            self.ui.setEnabled(True)

    @require_session
    def update_tci_inst(self):
        qi = self.session.tci.queued_instructions
        if len(qi) > 0:
            self.next_tci_inst = qi[0]
        else:
            self.next_tci_inst = None
            self.ui.l_tci_inst.setText('Next rate: --')
   
    @require_session
    def update_errors(self):
        errs = self.session.retrieve_errors()
        if len(errs) == 0:
            return
        errmsg = '\n'.join(errs)
        qtw.QMessageBox.critical(None, f'{len(errs)} errors/warnings', errmsg)
    
    @require_session
    def update_video(self):
        if self.ui.vm.isVisible():
            frame = self.session.cam.current_frame
            self.ui.vm.set_image(frame)
            
            audio = self.session.mic.get_current_audio()
            self.ui.vm.set_audio(audio)
            
    def update_spect_freq_selects(self, chan_name='', sel_idx=0, updated='txts'):

        sl_obj = self.ui.spect_freq_selects[f'{chan_name}'][sel_idx]
        txts = self.ui.spect_freq_select_valboxes[f'{chan_name}'][sel_idx]
        _lo, _hi = txts

        if updated == 'txts':
            _loval = _lo.text()
            _hival = _hi.text()
        elif updated == 'regions':
            _loval, _hival = sl_obj.getRegion()

        try:
            _loval = float(_loval)
        except:
            _loval = 0
        try:
            _hival = float(_hival)
        except:
            _hival = config.max_freq
        if _hival == 0:
            _hival = 1
        if _loval >= _hival:
            _loval = _hival-1

        _loval = np.round(_loval, 1)
        _hival = np.round(_hival, 1)

        sl_obj.setRegion([_loval, _hival])

        _lo.setText(str(_loval))
        _hi.setText(str(_hival))

        
    def end(self):
        for _, t in self.timers.items():
            t.stop()
        
        self.running = False

        if self.session is not None:
            self.session.end()
            self.session = None

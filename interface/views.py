from PyQt5 import QtWidgets as qtw
from PyQt5.QtCore import pyqtSignal, Qt, QSize, pyqtSlot, QSize
from PyQt5.QtGui import QPalette, QColor, QTransform, QFont, QImage, QPixmap
from PyQt5 import QtGui
import pyqtgraph as pg
import numpy as np
import warnings

from .checkable_combo_box import CheckableComboBox
import fus_anes.config as config
from fus_anes.util import sliding_window

pg.setConfigOption('imageAxisOrder', 'row-major')
pg.setConfigOption('useNumba', True)
sel_cols = [[0,0,0,255], [97,106,107,255], [189,195,199,255]]
chan_cols = ['#2b6bac', '#09c197', '#684ce7', '#d77915', '#a00b0b']
LR_margin = 3
YAxW = 30
cmap = pg.colormap.getFromMatplotlib(config.cmap)

class Interface(qtw.QWidget):
    closing = pyqtSignal()

    def __init__(self, app):

        super(Interface, self).__init__()
        self.app = app
        self.setWindowTitle('The Amazing Technicolor DreamCode')
        self.splash_screen = None
        self.main_layout = None
        self.setup()

        screen_size = self.app.primaryScreen().size()
        self.resize(screen_size.width()-50, screen_size.height()-100)

        self.show()
        
        self.vm = VideoMonitor()
        self.vm.show()
        self.vm.setVisible(False)

    def splash(self, on=False, message=None):
        if on==True and self.splash_screen is None:
            self.splash_screen = qtw.QSplashScreen(QtGui.QPixmap(config.loading_img_path).scaled(750,293),
                                                   Qt.WindowStaysOnTopHint)
            self.splash_screen.setWindowOpacity(0.9)
            self.splash_screen.show()
            if message:
                font = QFont()
                font.setBold(True)
                font.setPixelSize(20)
                self.splash_screen.setFont(font)
                self.splash_screen.showMessage(message, color=Qt.black, alignment=Qt.AlignCenter)
        elif on==False and self.splash_screen:
            self.splash_screen.close()
            self.splash_screen = None

    def _tl_plot(self, name, label_txt=None):
        tlayout = pg.GraphicsLayoutWidget()
        tlayout.setMinimumHeight(0)
        tlayout.ci.layout.setMinimumHeight(0)
        tlayout.setSizePolicy(qtw.QSizePolicy.Preferred, qtw.QSizePolicy.Ignored)
        tlayout.ci.layout.setContentsMargins(LR_margin,0,LR_margin,0)
        tlayout.setBackground('white')
        pw = tlayout.addPlot(row=0, col=0)

        pw.setMenuEnabled(False); pw.hideButtons()
            
        yaxis = pw.getAxis('left')
        yaxis.setWidth(YAxW)
        yaxis.setStyle(tickTextOffset=5, tickLength=3,
                       tickTextWidth=2, tickTextHeight=2)
        pw.showGrid(y=True, alpha=1.0)

        if label_txt is not None:
            vb = pw.getViewBox()
            ti = pg.TextItem(html=f'<div style="text-align: center"><span style="color: black;">{label_txt}</span></div>', anchor=(0,0))
            ti.setParentItem(pw)
            ti.setPos(35, 0)

        pw.setMouseEnabled(x=False, y=False)

        if 'timeline' in self.plot_objs:
            pw.setXLink(self.plot_objs['timeline'])
        if name == 'timeline':
            tlayout.setBackground([220, 220, 220])
            pw.showGrid(y=False, alpha=0)
            pw.showGrid(x=True, alpha=230)

        self.main_layout.addWidget(tlayout)
        self.frame_objs[name] = tlayout
        self.plot_objs[name] = pw
        return pw

    def setup_wv_frame(self, layout):

        class MultiLinePlot(pg.PlotWidget):
            def __init__(self, master_obj=None, **kwargs):
                self.master_obj = master_obj
                super().__init__(**kwargs)

            def wheelEvent(self, event):
                dy =  event.angleDelta().y() or event.pixelDelta().y()
                try:
                    gf = float(self.master_obj.t_zoomgain.text())
                except:
                    gf = 1.0
                    self.master_obj.t_zoomgain.setText(str(gf))
                dy = dy / gf
                self.master_obj.raw_eeg_zoom_factor += dy
                if self.master_obj.raw_eeg_zoom_factor < 0:
                    self.master_obj.raw_eeg_zoom_factor = 1e-8
            
        frame = qtw.QWidget()
        frame.setMinimumHeight(0)
        frame.setSizePolicy(qtw.QSizePolicy.Preferred, qtw.QSizePolicy.Ignored)
        
        flayout = qtw.QGridLayout()
        flayout.setSpacing(5)
        flayout.setContentsMargins(0,0,0,0)
        frame.setLayout(flayout)

        select = CheckableComboBox(elide=True) # which channels to display raw data
        init_selects = [1 for i in range(config.n_channels)]
        select.addItems([f'{i+1} ({n})' for i, n in zip(range(len(config.eeg_key)-1, -1, -1), config.eeg_key[::-1])],
                          initialStates=init_selects)
        select.updateText()
        self.eeg_raw_disp_select = select
        flayout.addWidget(select, 0, 0, 1, -1)
        flayout.setRowStretch(0, 1)
        
        self.t_lopass = qtw.QLineEdit(str(config.eeg_lopass))
        self.t_lopass.setFixedWidth(50)
        self.t_hipass = qtw.QLineEdit(str(config.eeg_hipass))
        self.t_hipass.setFixedWidth(50)
        self.t_notch = qtw.QLineEdit(str(config.eeg_notch))
        self.t_notch.setFixedWidth(50)
        self.t_zoomgain = qtw.QLineEdit(str(config.eeg_gain_zoom_factor))
        self.t_zoomgain.setFixedWidth(50)
        l0 = qtw.QLabel('High pass:')
        l1 = qtw.QLabel('Low pass:')
        l2 = qtw.QLabel('Notch:')
        l3 = qtw.QLabel('Fine zoom:')
        flayout.addWidget(l0, 1, 0, 1, 1, alignment=Qt.AlignRight)
        flayout.addWidget(self.t_hipass, 1, 1, 1, 1, alignment=Qt.AlignLeft)
        flayout.addWidget(l1, 1, 2, 1, 1, alignment=Qt.AlignRight)
        flayout.addWidget(self.t_lopass, 1, 3, 1, 1, alignment=Qt.AlignLeft)
        flayout.addWidget(l2, 1, 4, 1, 1, alignment=Qt.AlignRight)
        flayout.addWidget(self.t_notch, 1, 5, 1, 1, alignment=Qt.AlignLeft)
        flayout.addWidget(l3, 1, 6, 1, 1, alignment=Qt.AlignRight)
        flayout.addWidget(self.t_zoomgain, 1, 7, 1, 1, alignment=Qt.AlignLeft)
        flayout.setRowStretch(1, 1)

        pw = MultiLinePlot(master_obj=self)
        self.raw_eeg_zoom_factor = config.eeg_baseline_gain
        pw.setBackground('white')
        flayout.addWidget(pw, 2, 0, 1, -1)
        flayout.setRowStretch(2, 10)

        pw.setMouseEnabled(x=False, y=False)
        pw.enableAutoRange(enable=False)
        pw.setMenuEnabled(False); pw.hideButtons()

        yaxis = pw.getAxis('left')
        pw.setYRange(-1, config.n_channels+1, padding=0)
        yaxis.setTicks([[(c, f'{config.eeg_key[c]}') for c in np.arange(config.n_channels)[::-1]]])

        pw.setXRange(-0.05, config.raw_eeg_display_dur+0.05, padding=0)
        xaxis = pw.getAxis('bottom')
        xaxis.setTicks([[(t, f'{t:0.0f}') for t in np.arange(config.raw_eeg_display_dur+1)]])

        #yaxis.setWidth(YAxW)
        #yaxis.setStyle(tickTextOffset=5, tickLength=3,
        #               tickTextWidth=2, tickTextHeight=2)

        self.raw_panel = frame
        layout.addWidget(frame)
        layout.setStretchFactor(frame, 3) # size of EEG panel relative to main channels panel (which is 5)
        self.wv_obj = pw
        self.wv_plot_obj = None

    def setup(self):
        
        self.frame_objs = {}
        self.plot_objs = {}
        self.data_objs = {}
        self.channel_selects = {}
        
        self.master_layout = qtw.QVBoxLayout()
        self.master_layout.setContentsMargins(0,0,0,0)
        self.master_layout.setSpacing(5)
        self.setLayout(self.master_layout)

        # top toolbar
        tbar_frame = qtw.QFrame()
        tbar_frame.setFrameShape(qtw.QFrame.Box)
        tbar_frame.setLineWidth(3)
        tbar_frame.setStyleSheet("background-color: rgb(241, 255, 255); color: rgb(0, 0, 0);")
        tbar_layout = qtw.QVBoxLayout()
        tbar_layout.setSpacing(0)
        tbar_layout.setContentsMargins(0,0,0,0)
        tbar_frame.setLayout(tbar_layout)
        tbar_frame.setSizePolicy(qtw.QSizePolicy.Ignored, qtw.QSizePolicy.Preferred)

        # row 0 of top toolbar
        tbar_row0_layout = qtw.QHBoxLayout()
        tbar_row0_layout.setSpacing(0)
        tbar_row0_layout.setContentsMargins(0,3,0,20)
        # row 1 of top toolbar
        tbar_row1_layout = qtw.QHBoxLayout()
        tbar_row1_layout.setSpacing(0)
        tbar_row1_layout.setContentsMargins(0,0,0,3)

        self.b_sesh = qtw.QPushButton('New session')
        self.l_sesh = qtw.QLabel('(no session)')
        self.b_run_baseline = qtw.QPushButton('Baseline')
        self.b_run_squeeze = qtw.QPushButton('Squeeze')
        self.b_clear_tci_queue = qtw.QPushButton('Clear TCI')
        self.b_bolus = qtw.QPushButton('Bolus (mg)')
        self.t_bolus = qtw.QLineEdit('0')
        self.t_bolus.setFixedWidth(60)
        self.b_infusion = qtw.QPushButton('Infusion (mcg/kg/min)')
        self.t_infusion = qtw.QLineEdit('0')
        self.t_infusion.setFixedWidth(60)
        self.l_infusion_rate = qtw.QLabel('')
        self.b_simulate = qtw.QPushButton('Simulate')
        self.b_project = qtw.QPushButton('Project')
        self.b_set_tci_target = qtw.QPushButton('Goto TCI target')
        self.t_set_tci_target = qtw.QLineEdit('0')
        self.t_set_tci_target.setFixedWidth(60)
        self.b_marker = qtw.QPushButton('Mark')
        self.t_marker = qtw.QLineEdit('')
        self.b_lor = qtw.QPushButton('LOR')
        self.b_ror = qtw.QPushButton('ROR')
        self.b_reset_xlim = qtw.QPushButton('Reset timeline zoom/pan')
        self.b_reset_xlim.setEnabled(False)
        self.b_toggle_raw = qtw.QPushButton('Toggle main views')
        self.b_toggle_video = qtw.QPushButton('Toggle video/audio/CO2')

        row0_toolbar_items = [1, self.b_sesh, self.l_sesh, 50,
                             self.b_reset_xlim, 1,
                             self.b_toggle_raw, 1,
                             self.b_toggle_video,
                              50,
                              ]
        row1_toolbar_items = [1,
                              self.b_run_baseline, 5,
                              self.b_run_squeeze, 5,
                              self.b_marker, self.t_marker, 2,
                              self.b_lor, 1, self.b_ror, 1,
                              20,
                              self.b_bolus, self.t_bolus, 1,
                              self.b_infusion, self.t_infusion, 1, self.l_infusion_rate, 1,
                              self.b_simulate, 15,
                              self.b_project, 15,
                              self.b_set_tci_target, self.t_set_tci_target, 5,
                              self.b_clear_tci_queue, 5,
                              15,
                              ]

        for item in row0_toolbar_items:
            if isinstance(item, int):
                tbar_row0_layout.addStretch(item)
            else:
                tbar_row0_layout.addWidget(item)
        for item in row1_toolbar_items:
            if isinstance(item, int):
                tbar_row1_layout.addStretch(item)
            else:
                tbar_row1_layout.addWidget(item)

        tbar_layout.addLayout(tbar_row0_layout)
        tbar_layout.addLayout(tbar_row1_layout)

        self.master_layout.addWidget(tbar_frame)

        # lr frame/layout is all things below the top toolbar
        lr_frame = qtw.QWidget()
        lr_layout = qtw.QHBoxLayout()
        lr_layout.setContentsMargins(0,0,0,0)
        lr_layout.setSpacing(0)
        lr_frame.setLayout(lr_layout)
        self.master_layout.addWidget(lr_frame)
        
        # main frame/layout is all things below the top toolbar and left of the raw waveforms panel
        main_frame = qtw.QWidget()
        self.main_layout = qtw.QVBoxLayout()
        self.main_layout.setContentsMargins(5,5,5,5)
        self.main_layout.setSpacing(8)
        main_frame.setLayout(self.main_layout)
        self.chan_panel = main_frame
        lr_layout.addWidget(main_frame)
        lr_layout.setStretchFactor(main_frame, 5)

        # wv frame/layout is on the right where raw waveforms live
        self.setup_wv_frame(lr_layout)

        # timeline navigator (above timeline)
        pw = self._tl_plot('timeline_nav')
        xaxis = pw.getAxis('bottom')
        xaxis.setStyle(tickTextOffset=-18, tickLength=0)
        xaxis.setStyle(autoExpandTextSpace=False, autoReduceTextSpace=False)
        xaxis.setTicks([[]])
        yaxis = pw.getAxis('left')
        pw.setYRange(0, 1, padding=0)
        yaxis.setTicks([[]])
        pw.enableAutoRange(enable=False)
        pw.showGrid(y=False)
        lri = pg.LinearRegionItem(values=[0,1], orientation='vertical',
                                  pen=dict(width=0.001,),
                                  brush=(100, 100, 100, 120))
        pw.addItem(lri)
        v_bar = pg.InfiniteLine(pos=0, angle=90, pen=dict(color='black', width=3, alpha=0.5))
        pw.addItem(v_bar)
        pw.setMouseEnabled(x=True, y=False)
        self.timeline_nav_vbar = v_bar
        self.timeline_nav_lr = lri

        # timeline
        pw = self._tl_plot('timeline')
        xaxis = pw.getAxis('bottom')
        xaxis.setStyle(tickTextOffset=-18, tickLength=0)
        xaxis.setTextPen('black')
        xaxis.setStyle(autoExpandTextSpace=False, autoReduceTextSpace=False)
        v_bar = pg.InfiniteLine(pos=0, angle=90, pen=dict(color='white', width=5, alpha=0.5))
        pw.addItem(v_bar)
        self.data_objs[f'timeline'] = v_bar
        pw.setYRange(0, 1)
        yaxis = pw.getAxis('left')
        yaxis.setTicks([[]])
        pw.enableAutoRange(enable=False, axis='y')
        #pw.setMouseEnabled(x=True, y=False)
        self.markers_drawn = []

        # tci
        pw = self._tl_plot('tci', 'C<sub>e</sub> (<sup>&mu;g</sup>&frasl;<sub>mL</sub>)')
        pw.setYRange(0, config.tci_minval)
        pw.setMouseEnabled(x=False, y=True)
        lri = pg.LinearRegionItem(values=config.tci_display_target, orientation='horizontal',
                                  pen=dict(width=0.001,),
                                  brush=(255,236,144,120))
        yaxis = pw.getAxis('left')
        yaxis.setTicks([[(t, f'{t:0.1f}') for t in np.arange(0,3,0.1)]])
        pw.addItem(lri)
        self.data_objs[f'tci_hist'] = None
        self.data_objs[f'tci_sim'] = None
        self.data_objs[f'tci_proj'] = None
        self.bolusinfusion_markers_drawn = []
        
        # live channels
        self.spect_time_selects = {}
        self.spect_freq_selects = {}
        self.spect_freq_select_valboxes = {}
        self.slider_objs = []
        self.frame_objs['live_chs'] = []
        for disp_idx in range(config.n_live_chan):

            # frame for this live display
            frame = qtw.QWidget()
            layout = qtw.QGridLayout()
            layout.setContentsMargins(0,0,0,0)
            layout.setSpacing(0)
            layout.setHorizontalSpacing(20)
            frame.setLayout(layout)
            self.frame_objs['live_chs'].append(frame)

            # channel selection
            combobox = CheckableComboBox()#qtw.QComboBox()
            if disp_idx < len(config.eeg_init_chan_selects):
                init_selects = config.eeg_init_chan_selects[disp_idx]
            else:
                init_selects = config.eeg_init_chan_selects[-1]
            init_selects = [1 if i in init_selects else 0 for i in range(config.n_channels)]
            combobox.addItems([f'{i+1} ({n})' for i, n in enumerate(config.eeg_key)],
                              initialStates=init_selects)
            combobox.updateText()
            combobox.setStyleSheet(f'background-color: {chan_cols[disp_idx]};\
                                     selection-color:white;\
                                     selection-background-color:grey;\
                                     border-style:none;\
                                   ')
            self.channel_selects[disp_idx] = combobox
            #combobox.setFixedWidth(75)
            layout.addWidget(combobox, 0, 0, 1, 1)#, alignment=Qt.AlignLeft)

            # spect color scale
            self.slider_objs.append([])
            def cb(idx):
                do_name = f'eeg_spect_{idx}'
                pobj = self.data_objs.get(do_name, None)
                if pobj is None: return
                slo = self.slider_objs[idx][0].value()
                shi = self.slider_objs[idx][1].value()
                if shi == 0:
                    shi = 1
                    self.slider_objs[idx][1].setValue(shi)
                if slo >= shi:
                    slo = shi-1
                    self.slider_objs[idx][0].setValue(slo)
                pobj.setLevels([slo, shi])
            for lh in range(2):
                slider = qtw.QSlider(Qt.Horizontal)
                slider.setMinimum(0)
                slider.setMaximum(255)
                val = 255 if lh == 1 else 0
                slider.setValue(val)
                self.slider_objs[-1].append(slider)
                slider.sliderReleased.connect(lambda *_, idx=disp_idx: cb(idx))
                layout.addWidget(slider, 0, lh+1, 1, 1)
            
            # boxes for values of horizontal frequency band selections in spectrogram
            hfb_layout = qtw.QHBoxLayout()
            hfb_layout.setSpacing(5)
            hfb_layout.setContentsMargins(0,0,0,0)
            self.spect_freq_select_valboxes[f'eeg_spect_{disp_idx}'] = []
            for _i in range(config.n_spect_freq_selections):
                rang = config.spect_freq_defaults[_i]
                tlo = qtw.QLineEdit(str(rang[0]))
                thi = qtw.QLineEdit(str(rang[1]))
                tlo.setFixedWidth(35)
                thi.setFixedWidth(35)
                lab = qtw.QLabel(f'Band {_i+1}:')
                hfb_layout.addWidget(lab)
                hfb_layout.addWidget(tlo)
                hfb_layout.addWidget(thi)
                hfb_layout.addStretch(5)
                self.spect_freq_select_valboxes[f'eeg_spect_{disp_idx}'].append([tlo, thi])
            layout.addLayout(hfb_layout, 1, 0, 1, 1)
            
            layout.setColumnStretch(0, 3)
            layout.setColumnStretch(1, 1)
            layout.setColumnStretch(2, 1)
            
            # plots
            gframe = qtw.QWidget()
            glayout = qtw.QGridLayout()
            glayout.setContentsMargins(0,0,0,0)
            gframe.setSizePolicy(qtw.QSizePolicy.Preferred, qtw.QSizePolicy.Ignored)
            glayout.setSpacing(4)
            gframe.setLayout(glayout)

            def sub_plot():
                tlayout = pg.GraphicsLayoutWidget()
                tlayout.setMinimumHeight(0)
                tlayout.ci.layout.setMinimumHeight(0)
                tlayout.setSizePolicy(qtw.QSizePolicy.Preferred, qtw.QSizePolicy.Ignored)
                tlayout.setBackground('white')
                pw = tlayout.addPlot(row=0, col=0)
                pw.setMouseEnabled(x=False, y=False)
                return tlayout, pw

            # spectrogram
            tlayout, pw = sub_plot()
            self.plot_objs[f'eeg_spect_{disp_idx}'] = pw
            self.data_objs[f'eeg_spect_{disp_idx}'] = None
            glayout.addWidget(tlayout, 0, 0, 1, 2)

            tlayout.ci.layout.setContentsMargins(LR_margin,0,LR_margin,0)
            pw.setMenuEnabled(False); pw.hideButtons()
            yaxis = pw.getAxis('left')
            yaxis.setWidth(YAxW)
            yticks = np.arange(0, config.max_freq+1, 10)
            yaxis.setTicks([[(t, f'{t:0.0f}') for t in yticks]])
            pw.setYRange(0, config.max_freq, padding=0)
            pw.showGrid(y=True, alpha=1.0)
            #pw.setXLink(self.plot_objs['timeline']) # for some reason this alters alignment
            
            # spect time selections (vertical bars)
            objs = []
            for _i in range(config.n_spect_time_selections):
                sc = sel_cols[_i]
                sc_transp = sc.copy()
                sc_transp[-1] = 40
                v_bar = pg.LinearRegionItem(values=(20,25), movable=True, orientation='vertical',
                                        pen=dict(color=sc, width=2),
                                        brush=sc_transp)
                v_bar.setZValue(100)
                pw.addItem(v_bar)
                objs.append(v_bar)
            self.spect_time_selects[f'eeg_spect_{disp_idx}'] = objs
            
            # spect freq selections (horizontal bars)
            objs = []
            for _i in range(config.n_spect_freq_selections):
                rang = config.spect_freq_defaults[_i]
                h_bar = pg.LinearRegionItem(values=rang, movable=True, orientation='horizontal',
                                        pen=dict(color=[0,0,0,200], width=2),
                                        brush=[0,0,0,40])
                h_bar.setZValue(100)
                pw.addItem(h_bar)
                objs.append(h_bar)
            self.spect_freq_selects[f'eeg_spect_{disp_idx}'] = objs
            
            # psd
            tlayout, pw = sub_plot()
            pw.setMouseEnabled(x=True, y=True)

            tlayout.ci.layout.setContentsMargins(LR_margin,0,LR_margin,0)
            yaxis = pw.getAxis('left')
            yaxis.setWidth(YAxW)
            xaxis = pw.getAxis('bottom')
            xticks = np.arange(0, config.max_freq+1, 5)
            xaxis.setTicks([[(t, f'{t:0.0f}') for t in xticks]])
            pw.setXRange(0, config.max_freq, padding=0)
            pw.enableAutoRange(enable=False)
            pw.showGrid(x=True, y=True, alpha=0.5)

            self.plot_objs[f'eeg_psd_{disp_idx}'] = pw
            self.data_objs[f'eeg_psd_{disp_idx}'] = [None for _ in range(config.n_spect_time_selections)]
            glayout.addWidget(tlayout, 1, 0, 1, 2) # last 2->1 if readding another plot there

            layout.addWidget(gframe, 2, 0, 1, -1)
            self.main_layout.addWidget(frame)
            

        # layout final adjustments
        self.main_layout.setStretchFactor(tbar_frame, 1)
        self.main_layout.setStretchFactor(self.frame_objs['timeline_nav'], 1)
        self.main_layout.setStretchFactor(self.frame_objs['timeline'], 1)
        self.main_layout.setStretchFactor(self.frame_objs['tci'], 4)
        for frame in self.frame_objs['live_chs']:
            self.main_layout.setStretchFactor(frame, 10)

        # runtime
        self.timeline_xlim = [0, config.timeline_duration]
        self.auto_adjusting_timeline = False
        self.manually_active = False
        self.update_xticks()
        self.app.processEvents()
        self.update_timeline_nav(reset=True)
        self.plot_objs['timeline'].sigXRangeChanged.connect(self.user_zoompan)

    def get_chan_selections(self, disp_idx):
        obj = self.channel_selects[disp_idx].currentData()
        return [int(item.split(' ')[0]) - 1 for item in obj]
    
    def get_eeg_raw_disp_selections(self):
        obj = self.eeg_raw_disp_select.currentData()
        return [int(item.split(' ')[0]) - 1 for item in obj]

    def update_eeg_raw(self, xdata, ydata, vline):
        pobj = self.wv_obj
        objs = self.wv_plot_obj
        
        ydata = ydata - np.nanmean(ydata, axis=0)
        
        # determine selections for colouring
        cols = ['black' for i in range(config.n_channels)]
        cols[config.chan_reference] = 'grey'
        for disp_idx in range(config.n_live_chan):
            chan_idxs = self.get_chan_selections(disp_idx)
            for ci in chan_idxs:
                cols[ci] = chan_cols[disp_idx]

        # determine selections for display
        show = self.get_eeg_raw_disp_selections()

        if objs is None or len(show) != len(objs):

            # clear old lines if selections changed
            if objs is not None:
                pobj.removeItem(self.raw_eeg_vbar)
                for o in objs:
                    pobj.removeItem(o)

            objs = []
            for i in range(config.n_channels):
                if i not in show:
                    continue
                ydat = ydata[:, i] * self.raw_eeg_zoom_factor + i
                obj = pobj.plot(xdata, ydat, pen=cols[i])
                objs.append(obj)
            self.wv_plot_obj = objs
            self.raw_eeg_vbar = pg.InfiniteLine(pos=vline, movable=False, angle=90,
                                    pen=dict(color='black', width=3))
            pobj.addItem(self.raw_eeg_vbar)

            yaxis = pobj.getAxis('left')
            pobj.setYRange(-1, len(show)+1, padding=0)
            yaxis.setTicks([[(i, f'{config.eeg_key[c]}') for i,c in enumerate(show[::-1])]])
        else:
            self.raw_eeg_vbar.setValue(vline)
            for idx,ch in enumerate(show[::-1]):
                ydat = ydata[:, ch] * self.raw_eeg_zoom_factor + idx
                objs[idx].setData(xdata, ydat, pen=cols[ch])
        
    def update_eeg_spect(self, data, xvals, yvals):

        for disp_idx in range(config.n_live_chan):
            do_name = f'eeg_spect_{disp_idx}'
            obj = self.data_objs[do_name]
            pobj = self.plot_objs[do_name]

            chan_idxs = self.get_chan_selections(disp_idx)
            dat = data[chan_idxs]

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'All-NaN slice encountered')
                warnings.filterwarnings('ignore', r'Mean of empty slice')
                warnings.filterwarnings('ignore', r'invalid value encountered in subtract')
                warnings.filterwarnings('ignore', r'invalid value encountered in scalar subtract')
                dat = np.nanmean(dat, axis=0) # over selected channels
                minn, maxx = np.nanmin(dat), np.nanmax(dat)
                dat = 255 * (dat-minn) / (maxx-minn) # assume min will be zero
            dat[np.isnan(dat)] = 0
            dat = dat.astype(np.uint8)

            vmin = self.slider_objs[disp_idx][0].value()
            vmax = self.slider_objs[disp_idx][1].value()

            if obj is None:
                obj = pg.ImageItem(image=dat)
                obj.setColorMap(cmap)
                obj.setZValue(1)
                obj.setLevels([vmin, vmax])
                obj.setAutoDownsample(True)
                pobj.addItem(obj)
                self.data_objs[do_name] = obj
                pobj.setYRange(yvals[0], yvals[-1],
                               padding=0)

            else:
                obj.setImage(dat, autoLevels=False)
            
            tr = QTransform()
            xscale = (xvals[-1] - xvals[0]) / len(xvals)
            tr.scale(xscale,
                     (yvals[-1]-yvals[0]) / len(yvals))
            tr.translate(xvals[0] / xscale, 0)
            obj.setTransform(tr)
            
            pobj.setXRange(*self.timeline_xlim, padding=0)
            
    def update_tci_data(self, xvals, yvals, kind='hist'):
        # hist / sim / proj
        colors = dict(hist='black', proj='gray', sim='blue')
        col = colors[kind]
        zvals = dict(hist=100, sim=99, proj=98)
        zval = zvals[kind]

        pen = dict(color=col, width=3)
        if kind in ['proj', 'sim']:
            pen['style'] = Qt.DashLine

        data_name = f'tci_{kind}'

        pobj = self.plot_objs['tci']
        obj = self.data_objs[data_name]

        if obj is None:
            obj = pobj.plot(xvals, yvals, pen=pen)
            obj.setZValue(zval)
            pobj.enableAutoRange(enable=False)
            self.data_objs[data_name] = obj
        else:
            obj.setData(xvals, yvals)
        
        
        # commenting out for now but can reinstate if helpful
        # expand canvas if needed (Y)
        #minn, maxx = pobj.getViewBox().viewRange()[1]
        #nmax = np.nanmax(yvals)
        #if nmax!=0 and (nmax>maxx or nmax<maxx*0.5):
        #    y1 = max(nmax*1.2, config.tci_minval)
        #    pobj.setYRange(0, y1, padding=0)
       
        pobj.setXRange(*self.timeline_xlim, padding=0)
    
    def update_eeg_psd(self, disp_idx, selection_idx, data, xvals, **kw):
        do_name = f'eeg_psd_{disp_idx}'
        pobj = self.plot_objs[do_name]
        obj = self.data_objs[do_name][selection_idx]

        if obj is None:
            obj = pobj.plot(xvals, data, pen=dict(color=sel_cols[selection_idx], width=3))
            pobj.enableAutoRange(enable=True, axis='y')
            self.data_objs[do_name][selection_idx] = obj

        else:
            obj.setData(xvals, data)
            if np.all(data==0):
                obj.setVisible(False)
            else:
                obj.setVisible(True)

    def user_zoompan(self, view=None, xlim=None):
        if self.auto_adjusting_timeline:
            return
        self.manually_active = True
        self.b_reset_xlim.setEnabled(True)
        if view is not None:
            x0, x1 = view.viewRange()[0]
        elif view is None:
            x0, x1 = xlim
        self.timeline_xlim = [x0, x1]
        self.update_timeline_nav()
        
        # just because of the bug where spects cannot be x-linked without misaligning
        # so manually changing xlim of those ones each time we have an adjustment
        # this is done auto for tci and spect_trend bc they're linked to timeline
        pobjs = [p for p in self.plot_objs if p.startswith('eeg_spect_')]
        for pw in pobjs:
            self.plot_objs[pw].setXRange(*self.timeline_xlim, padding=0)

    def reset_xlim(self, t0, t1):
        self.manually_active = False
        self.b_reset_xlim.setEnabled(False)
        self.auto_adjusting_timeline = True
        self.timeline_xlim = [t0, t1]
        self.update_xticks()
        self.update_timeline_nav(reset=True)
        self.auto_adjusting_timeline = False
    
    def update_markers(self, markers=[]):
        pobj = self.plot_objs[f'timeline']

        for t in markers:
            if t in self.markers_drawn:
                continue
            v_bar = pg.InfiniteLine(pos=t, movable=False, angle=90,
                                    pen=dict(color='pink', width=2))
            pobj.addItem(v_bar)
            self.markers_drawn.append(t)

    def update_meds(self, boluses=[], infusions=[]):
        pobj = self.plot_objs[f'tci']

        cols = ['red']*len(boluses) + ['magenta']*len(infusions)
        meds = np.concatenate([boluses, infusions])
        for t, col in zip(meds, cols):
            if t in self.bolusinfusion_markers_drawn:
                continue
            v_bar = pg.InfiniteLine(pos=t, movable=False, angle=90,
                                    pen=dict(color=col, width=1))
            pobj.addItem(v_bar)
            self.bolusinfusion_markers_drawn.append(t)

    def update_timeline_nav(self, reset=False):
        pwn = self.plot_objs['timeline_nav']
        lro = self.timeline_nav_lr
        tlwidth = self.timeline_xlim[1] - self.timeline_xlim[0]
        cx0, cx1 = pwn.getAxis('bottom').range
        x0, x1 = cx0, cx1
        ax0 = self.timeline_xlim[0] - 0.25 * tlwidth
        ax1 = self.timeline_xlim[1] + 0.25 * tlwidth
        if self.timeline_xlim[0] <= cx0:
            x0, x1 = ax0, cx1
        if self.timeline_xlim[1] >= cx1:
            x0, x1 = cx0, ax1
        if reset:
            x0, x1 = ax0, ax1
        pwn.setXRange(x0, x1, padding=0)
        lro.setRegion([self.timeline_xlim[0], self.timeline_xlim[1]])

    def update_timeline(self, dt):
        # the main callback called very n milliseconds to keep timeline updated
        pobj = self.plot_objs[f'timeline']
        tl_obj = self.data_objs[f'timeline']
        tl_obj.setValue(dt)
        self.timeline_nav_vbar.setValue(dt)
        
        pobj.setXRange(*self.timeline_xlim, padding=0)
        
        # now move to shift the current view, but don't do anything if manual pan/zoom is active
        if self.manually_active:
            return
        
        if dt >= self.timeline_xlim[1]:
            new_x0 = self.timeline_xlim[0] + config.timeline_advance

            # update timeline
            self.timeline_xlim = [new_x0, new_x0 + config.timeline_duration]
            self.auto_adjusting_timeline = True
            self.update_xticks()
            self.update_timeline_nav(reset=True)
            self.auto_adjusting_timeline = False
            
    def update_xticks(self):
        pobjs = ['timeline', 'tci', 'timeline_nav'] +\
                [p for p in self.plot_objs if p.startswith('eeg_spect_')]

        xticks = np.arange(0, self.timeline_xlim[1]*10, 60)

        for p in pobjs:
            pw = self.plot_objs[p]
            pw.showGrid(x=True, alpha=0.75)
            xaxis = pw.getAxis('bottom')
            xaxis.setPen('black')
            pw.setXRange(*self.timeline_xlim, padding=0)
            if p =='timeline':
                '''
                if self.timeline_xlim[1] < 30*60:
                    modshow = 60*1, 10 # show text, show line, both in secs
                elif 30*60 <= self.timeline_xlim[1] < 60*60:
                    modshow = 60*5, 60
                elif self.timeline_xlim[1] >= 60*60:
                    modshow = 60*10, 60
                '''
                modshow = 60, 30 # show text, show line (secs)
                tlxticks = np.arange(0, self.timeline_xlim[1]*10, modshow[1])
                labs = [f'{x/60:0.0f}' if x%modshow[0]==0 else '' for x in tlxticks]
                xaxis.setTicks([[(t,l) for t,l in zip(tlxticks, labs)], [], []])
            elif p =='timeline_nav':
                modshow = 120, 30 # show text, show line (secs)
                tlxticks = np.arange(0, self.timeline_xlim[1]*10, modshow[1])
                labs = [f'{x/60:0.0f}' if x%modshow[0]==0 else '' for x in tlxticks]
                xaxis.setTicks([[(t,l) for t,l in zip(tlxticks, labs)], [], []])
            else:
                labs = ['']*len(xticks)
                xaxis.setTicks([[(t,l) for t,l in zip(xticks, labs)], [], []])
        

    def closeEvent(self, event):
        self.closing.emit() 


class VideoMonitor(qtw.QWidget):
    def __init__(self):
        super(VideoMonitor, self).__init__()
        
        self.layout = qtw.QVBoxLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setSpacing(5)
        
        self.resize(QSize(1440, 1000))
        
        self.size = QSize(1440, 810)
        self.label = qtw.QLabel(self)
        self.label.resize(self.size)
        
        tlayout = pg.GraphicsLayoutWidget()
        tlayout.setMinimumHeight(0)
        tlayout.ci.layout.setMinimumHeight(0)
        tlayout.setSizePolicy(qtw.QSizePolicy.Preferred, qtw.QSizePolicy.Ignored)
        tlayout.setBackground('white')
        
        # audio plot
        pw = tlayout.addPlot(row=0, col=0)
        pw.setMouseEnabled(x=False, y=False)
        adata = np.zeros(int(config.n_audio_display * 44100))[::10]
        xvals = np.arange(len(adata))
        self.a_plot_obj = pw.plot(xvals, adata, pen=dict(color='black'), clear=True)
        pw.setYRange(-15000, 15000)
        pw.enableAutoRange(enable=False)
        
        # co2 plot
        pw2 = tlayout.addPlot(row=1, col=0)
        pw2.setMouseEnabled(x=False, y=False)
        cdata = np.zeros(config.capnostream_live_buffer_length)
        xvals = np.arange(len(cdata))
        self.c_plot_obj = pw2.plot(xvals, cdata, pen=dict(color='black'), clear=True)
        pw2.setYRange(-1, 80)
        pw2.enableAutoRange(enable=False)
        
        self.layout.addWidget(tlayout)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        
        self.layout.setStretchFactor(tlayout, 3)
        self.layout.setStretchFactor(self.label, 5)

    def set_image(self, image):
        qImg = QImage(image, image.shape[1], image.shape[0], QImage.Format_RGB888)
        pm = QPixmap.fromImage(qImg)
        pm = pm.scaled(self.size, aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
        self.label.setPixmap(pm)

    def set_audio(self, data):
        data = data[::10]
        xvals = np.arange(len(data))
        self.a_plot_obj.setData(xvals, data)
        
    def set_co2(self, data):
        xvals = np.arange(len(data))
        self.c_plot_obj.setData(xvals, data)

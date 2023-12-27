import sys

import mne
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WaveletTypes, WaveletDenoisingTypes, \
    ThresholdTypes, WaveletExtensionTypes, NoiseEstimationLevelTypes, AggOperations
from mne import compute_raw_covariance
from mne.preprocessing import Xdawn
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np

class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title='BrainFlow Plot', size=(800, 600))

        self._init_timeseries()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtGui.QApplication.instance().exec_()

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot()
            print(curve)
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        # data = data[1:5]
        # eeg_channels = [0, 1, 2, 3]
        # sampling_rate = 256
        # for count, channel in enumerate(eeg_channels):
        #     DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
        #     DataFilter.perform_bandpass(data[channel], sampling_rate, 1.0, 45.0, 2,
        #                                 FilterTypes.BUTTERWORTH.value, 0)
        #     DataFilter.perform_bandstop(data[channel], sampling_rate, 48.0, 52.0, 2,
        #                                 FilterTypes.BUTTERWORTH.value, 0)
        #     DataFilter.perform_bandstop(data[channel], sampling_rate, 58.0, 62.0, 2,
        #                                 FilterTypes.BUTTERWORTH.value, 0)

        # data = data * 1e-6
        # ch_names = ['AF3', 'P9', 'P10', 'AF4']
        # ch_types = ['eeg', 'eeg', 'eeg', 'eeg']
        # sfreq = 256.0
        # info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        # raw = mne.io.RawArray(data, info)
        # raw.set_montage('standard_1020')
        # eog_channels = ["AF3", 'AF4']
        # raw.set_channel_types({ch: "eog" for ch in eog_channels})
        #
        # onset_fake = np.arange(start=0, stop=raw.times.max(), step=0.5)
        # duration_fake = 0.00001
        # marker_fake = ['fake_marker'] * len(onset_fake)
        # fake_annotation = mne.Annotations(onset=onset_fake,
        #                                   duration=duration_fake,
        #                                   description=marker_fake,
        #                                   orig_time=raw.info['meas_date'])
        # raw.set_annotations(fake_annotation)  # add to existing
        #
        # flat_criteria = dict(eeg=1e-6)
        # stronger_reject_criteria = dict(eeg=100e-6,  # 100 µV
        #                                 eog=100e-6)
        # events, event_dict = mne.events_from_annotations(raw)
        # event_dict_3 = {'fake_marker': 1}
        # epochs = mne.Epochs(raw, events, tmin=0, tmax=0.5,
        #                     reject=stronger_reject_criteria,
        #                     flat=flat_criteria,
        #                     event_id=event_dict_3, baseline=None, preload=True)
        # no_eog_channels = ["AF3", 'AF4']
        #
        # data = epochs.to_data_frame()
        # data = data[['AF3', 'AF4', 'P9', 'P10']].T
        # data = data.to_numpy()
        #
        # ch_names_ = [0, 1, 2, 3]

        # for count, channel in enumerate(ch_names_):
        #     print('')
        for count, channel in enumerate(self.exg_channels):
            # plot timeseries
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)

            self.curves[count].setData(data[channel].tolist())

        self.app.processEvents()


if __name__ == '__main__':
    params = BrainFlowInputParams()
    params.serial_port = "COM3"
    board = BoardShim(BoardIds.CYTON_BOARD, params)
    params.timeout = 25
    board.prepare_session()
    board.start_stream()
    # board.start_stream(450000, 'streaming_board://225.1.1.1:6677')
    # Graph(board)

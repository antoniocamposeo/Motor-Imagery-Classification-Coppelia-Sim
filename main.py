# import time
#
# import numpy as np
# from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
# from keras.models import model_from_json
#
#
# params = BrainFlowInputParams()
# params.serial_port = 'COM3'
# # params.ip_port = 6677
# # params.ip_address = "225.1.1.1"
# # params.master_board = BoardIds.CYTON_BOARD
# board = BoardShim(BoardIds.CYTON_BOARD, params)
# board.prepare_session()
# board.start_stream()
# for i in range(0,2):
#     time.sleep(5)
#     print(i)
#     data = board.get_board_data()
# board.stop_stream()
# board.release_session()
#
# data = data[1:4]
# data = data/1e-6
#
# import mne
# ch_names = ['C3','Cz','C4']
# ch_types = ['eeg']*3
# info = mne.create_info(ch_names=ch_names,
#                        sfreq=512, ch_types=ch_types
#                        )
# raw = mne.io.RawArray(data=data_1, info= info)
#
# import numpy as np
# np.save("data.npy",data)
#
#
#
#
#
#
# path_json = 'C:/Users/anto-/PycharmProjects/Motor_imagery_real_time/[ 32 300 128]_model.json'
# path_h5 = 'C:/Users/anto-/PycharmProjects/Motor_imagery_real_time/[ 32 300 128]_model.h5'
#
# # load json and create model
# json_file = open(path_json, 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights(path_h5)
#
# loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer="adam",
#                           metrics=['accuracy'])
#
#

from os import listdir
from os.path import isfile, join
import mne
import numpy as np
import pickle

#
# raw_file_train_2a = pickle.load(open("C:/Users/anto-/PycharmProjects/Motor_imagery_real_time/Data_Raw/raw_2a.dat", "rb"))
# raw_file_train_2b = pickle.load(open("C:/Users/anto-/PycharmProjects/Motor_imagery_real_time/Data_Raw/raw_2b.dat", "rb"))
#
# channels_drop_2a = ['EOG_1', 'EOG_0', 'EOG_2', 'EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2'
#     , 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-6', 'EEG-7', 'EEG-8'
#     , 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-15',
#                     'EEG-16', 'EEG-Pz']
#
# channels_drop_2b = ['EOG_1', 'EOG_0', 'EOG_2']
#
#
# for i in range(0,len(raw_file_train_2a)):
#     print(i)
#     raw = raw_file_train_2a[i]


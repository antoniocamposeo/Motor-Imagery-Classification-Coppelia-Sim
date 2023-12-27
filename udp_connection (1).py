import time
import numpy as np
import mne
import math
import pickle
import random
from pylsl import StreamInlet, resolve_stream
from time import sleep
from zmqRemoteApi import RemoteAPIClient


X_raw_eegnet = pickle.load(open("C:/Users/anto-/PycharmProjects/Motor_imagery_real_time/"
                                "Data_Raw/raw_X_eegnet.dat", "rb"))
Y_raw_eegnet = pickle.load(open("C:/Users/anto-/PycharmProjects/Motor_imagery_real_time/"
                                "Data_Raw/raw_Y_eegnet.dat", "rb"))

''' Connection Simulator '''
print('Program started')
# client = RemoteAPIClient()
# sim = client.getObject('sim')
# defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
# sim.setInt32Param(sim.intparam_idle_fps, 0)
#
# # Create a few dummies and set their positions:
# h = sim.getObjectHandle('/Manta')
# sim.setObjectPosition(h, -1, [0, 0, +0.3])
# sim.startSimulation()
# a = [0, 0]
# pack_input = sim.packFloatTable(a)
# sim.setStringSignal('signal', pack_input)
#
def print_direction(start_time):
    elapsed_time = time.time() - start_time
    if 1 <= elapsed_time < 4:
        print("-------------------------------------------------------")
    if elapsed_time >= 4:
        print("Risposo.")


''' Import pre-trained EEGNet CNN model '''
# Importing model
from keras.models import model_from_json

path_json = 'C:/Users/anto-/PycharmProjects/Motor_imagery_real_time/[ 32 500 128]_model.json'
path_h5 = 'C:/Users/anto-/PycharmProjects/Motor_imagery_real_time/[ 32 500 128]_model.h5'
# load json and create model
json_file = open(path_json, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(path_h5)

loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer="adam",
                     metrics=['accuracy'])


''' Classification with the imported EEGNet model '''
from sklearn.preprocessing import RobustScaler

fileObj = open('scaler[ 32 500 128].obj', 'rb')
sc = pickle.load(fileObj)  # scaler
fileObj.close()

''' Receive EEG stream from OpenBCI through Lab Streaming Layer protocol'''

# Connection to OpenBci

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

# duration acquisition
duration = 6
cross = 2
stop_cross = 3.25
sleep(1)

data = []
raw_array = []
raw_processed_array = []

totalNumSamples = 0  # count of samples
validSamples = 0  # count of valid samples
numChunks = 0  # number of blocks

print("Testing Sampling Rates...")

n_trial = 1  # number of trial
temp = 1
while True:
    # Acquisition
    start = time.time()
    print("Inizio Acquisizione Dati\n")
    print("Pensa di Muovere la mano destra o sinistra\n")
    while time.time() <= start + duration:
        #print_direction(start)
        # get chunks of samples
        chunk, timestamp = inlet.pull_chunk()

        if chunk:
            numChunks += 1
            # print(len(chunk))
            totalNumSamples += len(chunk)
            # print(chunk)
            for sample in chunk:
                print(sample)
                data.append(sample)
                validSamples += 1

    print("Stop Acquisizione Dati\n")
    sleep(2)
    print("Number of Chunks and Samples == {} , {}".format(numChunks, totalNumSamples))
    print("Valid Samples and Duration == {} / {}".format(validSamples, duration))
    print("Avg Sampling Rate == {}".format(validSamples / duration))
    data = np.array(data)

    # Extraction channels
    data = data.T
    data = data[:3, :1251]
    data = data * 5e-7  # scaling signal
    #data = X_raw_eegnet[random.randint(0,1848), :, :]
    ch_names = ['C3', 'Cz', 'C4']
    ch_types = ['eeg'] * 3
    info = mne.create_info(ch_names=ch_names,
                           sfreq=256, ch_types=ch_types
                           )
    raw = mne.io.RawArray(data=data, info=info)  # file raw
    raw_processed = raw.copy()  # copy of file raw
    raw_processed.filter(l_freq=8, h_freq=30, fir_design='firwin')  # file raw processed

    raw_array.append(raw)
    raw_processed_array.append(raw_processed)

    data_raw = raw_processed.get_data()
    data_raw = np.dstack([data_raw])
    kernels, chans, samples = 1, data_raw.shape[0], data_raw.shape[1]
    # kernels, chans, samples = 1, 3, 1251

    data_raw = data_raw.reshape(kernels, chans, samples)

    X_trial = sc.transform(data_raw.reshape(-1, data_raw.shape[-1])).reshape(data_raw.shape)

    prediction = loaded_model.predict(X_trial)
    print(f'Prediction Prob:{np.transpose(prediction)}')
    print(f"Prediction Value:{prediction.argmax(axis=-1)}")
    prediction_value = prediction.argmax(axis=-1)
    treshold = 0.7
    if prediction[0][1] >= 0.7:
        pred_val_t = 1
    else:
        pred_val_t = 0


    # if prediction_value == 1:
    #     sig = -math.pi / 8  # right
    # else:
    #     sig = math.pi / 8  # left
    #
    # sleep(0.4)
    #
    #
    # start = time.time()
    # a = [10,sig]
    # while time.time() - start < 1:
    #     pack_input = sim.packFloatTable(a)
    #     print('prova')
    #     sim.setStringSignal('signal', pack_input)
    #
    # a = [0,0]
    # pack_input = sim.packFloatTable(a)
    # sim.setStringSignal('signal', pack_input)

    sleep(0.5)
    f = open("test_1.txt", "a")
    f.write(f"numero prova:\t{temp}\n"
            f"-prob:[{prediction}]\n "
            f"-prediction:[{prediction_value}]\n"
            f"-pred con tresh :[{pred_val_t}]\n"
            )

    f.close()
    temp = temp +1
    print("STAT FERM")
    sleep(1)
    print("TRA 2 SECONDI INIZI ")
    sleep(2)

    data = []
#
# pickle.dump(raw_array, open(
#     'C:/Users/zenzo/OneDrive - Politecnico di Bari/Materiale UNI/MAGISTRALE/Artificial Intelligence & Machine Learning/Project/Motor_imagery_real_time/raw_array.dat',
#     'wb'))
# pickle.dump(raw_processed_array, open(
#     'C:/Users/zenzo/OneDrive - Politecnico di Bari/Materiale UNI/MAGISTRALE/Artificial Intelligence & Machine Learning/Project/Motor_imagery_real_time/raw_processed_array.dat',
#     'wb'))

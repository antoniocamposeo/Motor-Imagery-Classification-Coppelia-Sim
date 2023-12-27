import time
import numpy as np
from pylsl import StreamInlet, resolve_stream
from time import sleep
import mne

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
import pickle
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
sleep(1)

data = []


totalNumSamples = 0  # count of samples
validSamples = 0  # count of valid samples
numChunks = 0  # number of blocks

print("Testing Sampling Rates...")


n_trial = 1  # number of trial

while True:
    # Acquisition
    start = time.time()
    while time.time() <= start + duration:
        # get chunks of samples
        chunk, timestamp = inlet.pull_chunk()

        if chunk:
            numChunks += 1
            print(len(chunk))
            totalNumSamples += len(chunk)
            print(chunk)
            for sample in chunk:
                print(sample)
                data.append(sample)
                validSamples += 1

    print("Number of Chunks and Samples == {} , {}".format(numChunks, totalNumSamples))
    print("Valid Samples and Duration == {} / {}".format(validSamples, duration))
    print("Avg Sampling Rate == {}".format(validSamples / duration))
    data = np.array(data)
    print(data.shape)

    # Extraction channels
    #data = data[0:3]
    data = data.T
    data = data[:3, :1251]
    data = data * 1e-6  # scaling signal
    ch_names = ['C3', 'Cz', 'C4']
    ch_types = ['eeg'] * 3
    info = mne.create_info(ch_names=ch_names,
                       sfreq=256, ch_types=ch_types
                       )
    raw = mne.io.RawArray(data=data, info=info) # file raw
    raw_processed = raw.copy()  # copy of file raw
    raw_processed.filter(l_freq=8, h_freq=30, fir_design='firwin') # file raw processed

    data_raw = raw_processed.get_data()
    data_raw = np.dstack([data_raw])
    kernels, chans, samples = 1, data_raw.shape[0], data_raw.shape[1]
    #kernels, chans, samples = 1, 3, 1251

    data_raw = data_raw.reshape(kernels, chans, samples)

    X_trial = sc.transform(data_raw.reshape(-1, data_raw.shape[-1])).reshape(data_raw.shape)

    prediction = loaded_model.predict(X_trial)

    print(f"Prediction Value:{prediction.argmax(axis=-1)}")
    data = []
    time.sleep(1)

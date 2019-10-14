import os

import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras

import random
import librosa


train_audio_path = 'C:/tmp/speech_dataset/'
folders = next(os.walk(train_audio_path))[1] # 0=se stesso, 1=subfolder primo livello ...
folders.sort()
print(f"All available folders: {folders}") 

target_list = ['backward', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
unknown_list = list(filter(lambda f: f not in target_list and f != '_background_noise_', folders))
print(f'target_list : {target_list}, unknowns: {unknown_list}, no_word: [\'_background_noise_\']')

x = [] # lista di wav dei comandi a cui siamo interessati (arrays of 8000 floats)
y = [] # lista di label associati ai wav qui sopra: ['no', 'no', 'no', 'yes', 'yes', 'on' ....]

# andiamo a leggere i files con resampling 8000Hz per velocizzare e diminuire l'uso della memoria

unknown_wav = [] # lista di wav a cui non siamo interessati
y_value_map = {} # dizionario chiave-valore: { 'no': 1, 'yes': 2, 'up': 3 .... }
background_noise = []



for idx, folder in enumerate(folders): #ciclo tutte le sottocartelle:
    print(f"Loading folder {folder}")
    if folder !=  '_background_noise_':
        y_value_map[folder] = idx

    for wav_file in filter(lambda f: f.endswith('.wav'), os.listdir(os.path.join(train_audio_path, folder))):
        wav_file_full_path = os.path.join(train_audio_path, folder,  wav_file)
        waf_file_8Hz_full_path = wav_file_full_path + '.npy'

        if not os.path.exists(waf_file_8Hz_full_path):
            samples, sample_rate = librosa.load(wav_file_full_path) # load original file (original 16KHz)
            sample_8Hz = librosa.resample(samples, sample_rate, 8000)
            np.save(waf_file_8Hz_full_path, sample_8Hz)
        else: 
            sample_8Hz = np.load(waf_file_8Hz_full_path)

        if folder == '_background_noise_':
            if len(sample_8Hz) >= 8000:
                background_noise.append(sample_8Hz)
        elif folder in unknown_list:
            if len(sample_8Hz) == 8000:
                unknown_wav.append(sample_8Hz)
        else:
            if len(sample_8Hz) == 8000:
                y.append([folder])
                x.append(sample_8Hz)       

# noise files are > 1 seconds
# take randomly 1 seconds of noise
def get_random_noise(noise_num = 0):
    selected_noise = background_noise[noise_num]
    start_idx = random.randint(0, len(selected_noise)- 1 - 8000)
    return selected_noise[start_idx:(start_idx + 8000)]
    
X = np.stack(x).reshape(-1, 8000, 1)
Y = np.array(y)
    
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=666, shuffle=True)

# Parameters
lr = 0.001
generations = 20000
num_gens_to_wait = 250
batch_size = 512
drop_out_rate = 0.5
input_shape = (8000, 1)

#For Conv1D add Channel
X_train = X_train.reshape(-1,8000,1)
X_test = X_test.reshape(-1,8000,1)

#Make Label data 'class num' -> 'One hot vector'
y_train = keras.utils.to_categorical(y_train, len(y_value_map))
y_test = keras.utils.to_categorical(y_test, len(y_value_map))

#Conv1D Model
model = keras.models.Sequential([
    keras.layers.Conv1D(filters=8, kernel_size=11, padding='valid', activation='relu', strides=1, input_shape=input_shape),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Dropout(drop_out_rate),
    keras.layers.Conv1D(16, 7, padding='valid', activation='relu', strides=1),
    keras.layers.MaxPooling1D(2),
    keras.layers.Dropout(drop_out_rate),
    keras.layers.Conv1D(32, 5, padding='valid', activation='relu', strides=1),
    keras.layers.MaxPooling1D(2),
    keras.layers.Dropout(drop_out_rate),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(drop_out_rate),
    keras.layers.Dense(len(y_value_map), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['sparse_categorical_accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=20)
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Loss: {loss}, accuracy: {accuracy}")
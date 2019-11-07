# mfcc -->  https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd
#           http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
# CNN  -->  https://www.jessicayung.com/explaining-tensorflow-code-for-a-convolutional-neural-network/
#           https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/

# ISTRUZIONI PER FARE TRAIN DEL MODELLO
# 1) Installare python 3.7.x (tensorflow 2 non supporta ancora il 3.8)
# 2) aprire il terminale ed eseguire i seguenti (spero siano tutti):
#    - pip install numpy
#    - pip install scikit-learn
#    - pip install tensorflow (oppure tensorflow-gpu se hai una scheda video compatibile)
#    - pip install librosa
#    - pip install matplotlib
# 3) scaricare il training set di google https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
# 4) scompattarlo in C:/tmp/speech_dataset/
# 5) eseguire Train.py
#
# -> se tutto va bene, verrà creato un file "saving/T2.tflite" contenente il modello trainato e "savinc/T2_mapping.json" per capire l'output

import os
import datetime

import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras

import random
import librosa
import librosa.display
import json

# path che contiene il set di file audio
train_audio_path = 'C:/tmp/speech_dataset/'

# quali parole voglio riconoscere
target_list = ['backward', 'yes', 'no', 'on', 'off','up', 'down', 'stop', 'go', 'right', 'forward','left']
# parole che non voglio processare
skip_list = ['tree', 'one', 'bed', 'eight', 'three', 'two', 'bird', 'cat', 'dog', 'five', 'four', 'seven', 'sheila', 'six', 'follow', 'visual', 'zero', 'marvin', 'nine', 'wow', 'happy', 'learn', 'house']


folders = next(os.walk(train_audio_path))[1] # 0=se stesso, 1=subfolder primo livello ...
folders = list(filter(lambda f: f not in skip_list, folders))
folders.sort()
print(f"All available folders: {folders}") 

unknown_list = list(filter(lambda f: f not in target_list and f != '_background_noise_', folders))
print(f'target_list : {target_list}, unknowns: {unknown_list}, no_word: [\'_background_noise_\']')

x = [] # lista di wav dei comandi a cui siamo interessati (arrays of 8000 floats)
y = [] # lista di label associati ai wav qui sopra: ['no', 'no', 'no', 'yes', 'yes', 'on' ....]

#unknown_wav = [] # lista di wav a cui non siamo interessati
y_value_map = {} # dizionario chiave-valore: { 'no': 1, 'yes': 2, 'up': 3 .... }
background_noise = []

idx = 0
y_value_map['_unknown_'] = idx

for folder in folders: #ciclo tutte le sottocartelle (a cui sono interessato):
    print(f"Loading folder {folder}")
    if folder !=  '_background_noise_' and folder not in unknown_list:
        idx += 1
        y_value_map[folder] = idx

    # ciclo tutti i file .wav
    for wav_file in filter(lambda f: f.endswith('.wav'), os.listdir(os.path.join(train_audio_path, folder))):
        wav_file_full_path = os.path.join(train_audio_path, folder,  wav_file) # full path file wav
        waf_file_spectogram_full_path = wav_file_full_path + '.spec.npy' # salvo un file lo spettrogramma come array numpy (facile rileggere)

        # creo spettrogramma se non avevo già calcolato in precedenza, altrimenti lo carica da file
        if not os.path.exists(waf_file_spectogram_full_path):
            samples, sample_rate = librosa.load(wav_file_full_path, sr=16000) # load original file (original 16KHz)
            spectogram = librosa.feature.melspectrogram(y=samples, sr=sample_rate) # calcola spettrogramma a bin
            np.save(waf_file_spectogram_full_path, spectogram)            
        else: 
            spectogram = np.load(waf_file_spectogram_full_path)

        if (50 > spectogram.shape[1]): # padding con zero
            pad_width = 50 - spectogram.shape[1]
            spectogram = np.pad(spectogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
        if(spectogram.shape[1] > 50):
            continue

        if folder == '_background_noise_':
                background_noise.append(spectogram) # not used now
        elif folder in unknown_list:
                #unknown_wav.append(sample_8Hz)
                y.append('_unknown_')
                x.append(spectogram)       
        else:
            y.append(folder)
            x.append(spectogram)       

max_x = max(map(lambda n: n.shape[1], x))

X = np.stack(x)
Y = np.array(y)
    
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=666, shuffle=True)

input_shape = X_train[0].shape # 128, 50

X_train = np.reshape(X_train, (-1, input_shape[0], input_shape[1], 1))
X_test = np.reshape(X_test, (-1, input_shape[0], input_shape[1], 1))

input_shape = X_train[0].shape # 128, 50, 1

#Make Label data 'class num' -> 'One hot vector' 1 -> [1,0,0,0,0,0], 2 -> [0,1,0,0,0,0,0]...
y_train = np.vectorize(y_value_map.get)(y_train)
y_test = np.vectorize(y_value_map.get)(y_test)
y_train = keras.utils.to_categorical(y_train, len(y_value_map))
y_test = keras.utils.to_categorical(y_test, len(y_value_map))

#Conv1D Model
model = keras.models.Sequential([
    keras.layers.Conv2D(
        filters=32, # numero filtri
        kernel_size=(3,3), # dimensione filtro/kernel
        strides=(1, 1),
        padding='same', #add padding
        activation='relu',
        input_shape=input_shape
    ),
    keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding='valid'),
    keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding='valid'),
    keras.layers.Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding='valid'),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(), # da immagine 2D ad array
    keras.layers.Dense(
        units=64,
        activation='relu'
    ),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(
        units = y_train.shape[1],
        activation='softmax'
    )
])

model.compile(loss='categorical_crossentropy',
             optimizer=tf.optimizers.Adam(learning_rate=0.01),
             metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=10) #effettua il training (ha senso aumentare le epoche)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, accuracy: {accuracy}") # stampa performance su validation set


# salva il modello
tf.saved_model.save(model, "./saving/T2_savedModel/") 

# converte SavedModel a tensorflow lite
converter = tf.lite.TFLiteConverter.from_saved_model("./saving/T2_savedModel/")
with open("./saving/T2.tflite", "wb") as tffile:
    tffile.write(converter.convert())

# salva mapping per capire a runtime l'output
json.dump( y_value_map, open( "./saving/T2_mapping.json", 'w' ) )
# mfcc -->  https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd
#           http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
# CNN  -->  https://www.jessicayung.com/explaining-tensorflow-code-for-a-convolutional-neural-network/
#           https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/

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


train_audio_path = 'C:/tmp/speech_dataset/'

target_list = ['backward', 'yes', 'no', 'on', 'off']
skip_list = ['tree', 'one', 'bed', 'eight', 'three', 'two','up', 'down', 'stop', 'go', 'left', 'right', 'bird', 'cat', 'dog', 'five', 'four', 'seven', 'sheila', 'six', 'follow', 'visual', 'zero', 'marvin', 'nine', 'wow', 'happy', 'forward', 'learn', 'house']

#target_list = ['backward', 'yes', 'no', 'on', 'off','up', 'down', 'stop', 'go', 'right', 'forward','left']
#skip_list = ['tree', 'one', 'bed', 'eight', 'three', 'two', 'bird', 'cat', 'dog', 'five', 'four', 'seven', 'sheila', 'six', 'follow', 'visual', 'zero', 'marvin', 'nine', 'wow', 'happy', 'learn', 'house']


folders = next(os.walk(train_audio_path))[1] # 0=se stesso, 1=subfolder primo livello ...
folders = list(filter(lambda f: f not in skip_list, folders))
folders.sort()
print(f"All available folders: {folders}") 

unknown_list = list(filter(lambda f: f not in target_list and f != '_background_noise_', folders))
print(f'target_list : {target_list}, unknowns: {unknown_list}, no_word: [\'_background_noise_\']')

x = [] # lista di wav dei comandi a cui siamo interessati (arrays of 8000 floats)
y = [] # lista di label associati ai wav qui sopra: ['no', 'no', 'no', 'yes', 'yes', 'on' ....]

# andiamo a leggere i files con resampling 8000Hz per velocizzare e diminuire l'uso della memoria

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

    for wav_file in filter(lambda f: f.endswith('.wav'), os.listdir(os.path.join(train_audio_path, folder))):
        wav_file_full_path = os.path.join(train_audio_path, folder,  wav_file)
        wav_file_image_full_path = wav_file_full_path + '.png'
        waf_file_spectogram_image_full_path = wav_file_full_path + '.spec.png'
        waf_file_spectogram_full_path = wav_file_full_path + '.spec.npy'
        waf_file_mfcc_image_full_path = wav_file_full_path + '.mfcc.png'
        waf_file_mfcc_full_path = wav_file_full_path + '.mfcc.npy'

        if not os.path.exists(waf_file_spectogram_full_path):
            samples, sample_rate = librosa.load(wav_file_full_path, sr=16000) # load original file (original 16KHz)
            spectogram = librosa.feature.melspectrogram(y=samples, sr=sample_rate)

            mfccs = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=40)
            #mfccsscaled = np.mean(mfccs.T,axis=0)

            # WAVEP
            # save waveplot as a file
            # fig = plt.figure(figsize=[2, 1])
            # ax = fig.add_subplot(111)
            # ax.axes.get_xaxis().set_visible(False)
            # ax.axes.get_yaxis().set_visible(False)
            # ax.set_frame_on(False)
            # librosa.display.waveplot(samples, sr=sample_rate)
            # plt.savefig(wav_file_image_full_path, dpi=400, bbox_inches='tight', pad_inches=0)
            # plt.close('all')
            #SPECTOGRAM
            # save raw spectogram
            np.save(waf_file_spectogram_full_path, spectogram)
            # save spectogram as a file
            # fig = plt.figure(figsize=[2, 1])
            # ax = fig.add_subplot(111)
            # ax.axes.get_xaxis().set_visible(False)
            # ax.axes.get_yaxis().set_visible(False)
            # ax.set_frame_on(False)
            # librosa.display.specshow(librosa.power_to_db(spectogram, ref=np.max))
            # plt.savefig(waf_file_spectogram_image_full_path, dpi=400, bbox_inches='tight', pad_inches=0)
            # plt.close('all')
            
            #MFCC
            # save raw spectogram
            np.save(waf_file_mfcc_full_path, mfccs)
            # save spectogram as a file
            # fig = plt.figure(figsize=[2, 1])
            # ax = fig.add_subplot(111)
            # ax.axes.get_xaxis().set_visible(False)
            # ax.axes.get_yaxis().set_visible(False)
            # ax.set_frame_on(False)
            # librosa.display.specshow(mfccs, sr=sample_rate)
            # plt.savefig(waf_file_mfcc_image_full_path, dpi=400, bbox_inches='tight', pad_inches=0)
            # plt.close('all')
            
        else: 
            spectogram = np.load(waf_file_spectogram_full_path)


        if (50 > spectogram.shape[1]): # pad to have same width
            pad_width = 50 - spectogram.shape[1]
            spectogram = np.pad(spectogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
        if(spectogram.shape[1] > 50):
            continue

        if folder == '_background_noise_':
                background_noise.append(spectogram)
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

# Parameters
class LearningRateReducerCb(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    old_lr = self.model.optimizer.lr.read_value()
    new_lr = old_lr * 0.99
    print(f"\nEpoch: {epoch}. Reducing Learning Rate from {old_lr} to {new_lr}")
    self.model.optimizer.lr.assign(new_lr)

drop_out_rate = 0.25


#Make Label data 'class num' -> 'One hot vector' 1 -> [1,0,0,0,0,0], 2 -> [0,1,0,0,0,0,0]...
y_train = np.vectorize(y_value_map.get)(y_train)
y_test = np.vectorize(y_value_map.get)(y_test)
y_train = keras.utils.to_categorical(y_train, len(y_value_map))
y_test = keras.utils.to_categorical(y_test, len(y_value_map))

#Conv1D Model
model = keras.models.Sequential([
    keras.layers.Conv2D(
        filters=32,
        kernel_size=(3,3), #size of filter
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
    keras.layers.Dropout(drop_out_rate),
    keras.layers.Flatten(),
    keras.layers.Dense(
        units=64,
        activation='relu'
    ),
    keras.layers.Dropout(drop_out_rate),
    keras.layers.Dense(
        units = y_train.shape[1],
        activation='softmax'
    )
])

model.compile(loss='categorical_crossentropy',
             optimizer=tf.optimizers.Adam(learning_rate=0.01),
             #batch_size=512,
             metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, callbacks=[LearningRateReducerCb()], epochs=10)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, accuracy: {accuracy}")

tf.saved_model.save(model, "./saving/T2_savedModel/") 

# convert SavedModel to tensorflow lite model
converter = tf.lite.TFLiteConverter.from_saved_model("./saving/T2_savedModel/")
with open("./saving/T2.tflite", "wb") as tffile:
    tffile.write(converter.convert())

# save mapping to understand output
json.dump( y_value_map, open( "./saving/T2_mapping.json", 'w' ) )
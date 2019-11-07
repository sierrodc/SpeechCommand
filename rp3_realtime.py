# ISTRUZIONI
# 1) installare python 3.7.x
# 2) installare i seguenti pacchetti
#	 - sudo apt-get install llvm
#	 - pip3 install llvmlite
#	 - pip3 install librosa
#	 - sudo apt-get install python-pyaudio python3-pyaudio
#	 - pip3 install pyaudio
#	 - sudo apt-get install alsa-tools alsa-utils
#	 - alsactl init
#	 - sudo apt-get install portaudio19-dev
#    - installare tensorflow lite da qui: https://www.tensorflow.org/lite/guide/python (scaricare il whl compatibile)
# 3) impostare il microfono usb come dispositivo di default (menu -> preferences -> audio device settings)
# 4) lanciare questo programma

import numpy as np
import pyaudio
import time
import librosa
# import tensorflow as tf
import tflite_runtime.interpreter as tflite
import json

class AudioStreamRecognizer(object):
    def __init__(self):
		
		# carica il modello
        self.interpreter = tflite.Interpreter(model_path="./saving/T2.tflite")
        self.interpreter.allocate_tensors()
		# carica il mapping (output modello con labels)
        y_value_map = json.load( open( "./saving/T2_mapping.json" ) )
        self.inv_map = {v: k for k, v in y_value_map.items()}
		
        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape']
		
        self.RATE = 16000
        self.ringBuffer = np.zeros(self.RATE, dtype=float) # crea un buffer (vuoto) di 16000 numeri
        # chunk = numero di samples per volta. se == rate allora 1 chunk al secondo
        self.CHUNK = int(self.RATE / 8) # ogni 1/8 di secondo

        # stream object. Verrà chiamato callback ogni volta che sarà disponibile il chunk (=> ogni 1/8 di secondo)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels= 1,
            rate=self.RATE,
            input=True,
            # output=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.callback
        )

    def start_listening(self):
        self.stream.start_stream()

    def is_listening(self):
        return self.stream.is_active()
    
    def callback(self, in_data, frame_count, time_info, flag):
        start_time = time.perf_counter() 
		
		# recupera il nuovo audio
        audio_data = np.fromstring(in_data, dtype=np.float32)
		
        # sposta a sinistra gli elementi e aggiunge quelli nuovi
        len_audioData = len(audio_data)
        self.ringBuffer = np.roll(self.ringBuffer, -len_audioData)
        np.put(self.ringBuffer, range(self.RATE-len_audioData, self.RATE), audio_data)

		# crea lo spettrogramma (stesso modo in cui veniva fatto nel training)
        spectogram = librosa.feature.melspectrogram(y=self.ringBuffer, sr=self.RATE)
        if (50 > spectogram.shape[1]): # pad to have same width
            pad_width = 50 - spectogram.shape[1]
            spectogram = np.pad(spectogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
        spectogram = np.reshape(spectogram, self.input_shape)

        self.interpreter.set_tensor(self.input_details[0]['index'], spectogram)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        maxValue = np.max(output_data)
        maxValueIndex = np.argmax(output_data)

        if (maxValue>0.80):
            print(f"Predicted '{self.inv_map[maxValueIndex]}' with probability={maxValue*100:2f}")
            # clear buffer:
            self.ringBuffer = np.zeros(self.RATE, dtype=float)

        print(f'Everything done in {(time.perf_counter() -start_time)*1000} ms')

        return (in_data, pyaudio.paContinue)



if __name__ == '__main__':
    audioStreamRecognizer = AudioStreamRecognizer()
    audioStreamRecognizer.start_listening()
    while audioStreamRecognizer.is_listening():
        time.sleep(0.25)
    
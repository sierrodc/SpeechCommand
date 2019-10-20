import numpy as np
import tensorflow as tf
###### import tflite_runtime.interpreter as tflite
import json
import librosa
import sys

if len(sys.argv) < 2:
    print(f"Usage: python3 {sys.argv[0]} filepath.wav")
    sys.exit()

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./saving/T2.tflite")
###### interpreter = tflite.Interpreter(model_path="./saving/T2.tflite")
interpreter.allocate_tensors()

y_value_map = json.load( open( "./saving/T2_mapping.json" ) )
inv_map = {v: k for k, v in y_value_map.items()}

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run model on random input data.
input_shape = input_details[0]['shape']

samples, sample_rate = librosa.load(sys.argv[1], sr=16000) # load original file (original 16KHz)
spectogram = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
if (50 > spectogram.shape[1]): # pad to have same width
    pad_width = 50 - spectogram.shape[1]
    spectogram = np.pad(spectogram, pad_width=((0, 0), (0, pad_width)), mode='constant')

spectogram = np.reshape(spectogram, input_shape)

interpreter.set_tensor(input_details[0]['index'], spectogram)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

maxValue = np.max(output_data)
maxValueIndex = np.argmax(output_data)

print(f"Predicted '{inv_map[maxValueIndex]}' with probability={maxValue*100:2f}")

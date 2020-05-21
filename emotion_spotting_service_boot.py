import tensorflow.keras as keras
from tensorflow.keras.models import model_from_json

from tensorflow.keras import losses
import numpy as np
import librosa
import tensorflow as tf
import pickle

MODEL_PATH = './mfcc_aug_model.h5'
JSON_MODEL_PATH = './model_test.json'
JSON_MODEL_WEIGHTS = './model_test.h5'
LABEL_ENC_PICKLE_PATH = './labels_mfcc_aug'
NUM_SAMPLES_TO_CONSIDER = 22050 * 5  # 5sec


class _Emotion_Spotting_Service:  # singleton class

    model = None  # tensorflow model

    filename = LABEL_ENC_PICKLE_PATH
    infile = open(filename, 'rb')
    labels = pickle.load(infile)
    infile.close()

    _mappings = labels.classes_.tolist()

    _instance = None  # this is a trick to implement the class as a simpleton.

    def predict(self, file_path):
        # extract MFCCs
        MFCCs = self.preprocess(file_path)  # (# segments, # coefficients)

        # convert 2D MFCCs into 4D array -> (#samples, #segments, #mfccs, depth=1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make prediction
        predictions = self.model.predict(MFCCs)  # [ [0.1, 0.6, 0.1, ...] ]
        predicted_index = np.argmax(predictions)  # returns index of highest value in array
        predicted_emotion = self._mappings[predicted_index]

        # separate gender and emotion from prediction
        pred_list = predicted_emotion.split('_')
        gender = pred_list[0]
        emotion = pred_list[1]

        return gender, emotion

    def preprocess(self, file_path, n_mfcc=30, n_fft=2048, hop_length=512):
        # load audio file
        signal, sr = librosa.load(file_path)

        # ensure consistency in the audio file length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        sample_rate = np.array(sr)

        # Random offset. If signal > expected samples, randomly offset and pick signal of expected len
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            max_offset = len(signal) - NUM_SAMPLES_TO_CONSIDER
            offset = np.random.randint(max_offset)
            signal = signal[offset:(NUM_SAMPLES_TO_CONSIDER + offset)]

        # Random padding. If expected samples > signal, pad signal.
        else:
            if NUM_SAMPLES_TO_CONSIDER > len(signal):
                max_offset = NUM_SAMPLES_TO_CONSIDER - len(signal)
                offset = np.random.randint(max_offset)
                # offset = int(max_offset/2)

            else:
                max_offset = 0
                offset = 0

            signal = np.pad(signal, (offset, max_offset - offset), "constant")

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        return MFCCs

    # def get_2d_conv_model(self,n):


def Emotion_Spotting_Service():
    # ensure that we only have 1 instance of KSS
    if _Emotion_Spotting_Service._instance is None:
        _Emotion_Spotting_Service._instance = _Emotion_Spotting_Service()

        # load model form json
        json_file = open(JSON_MODEL_PATH, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        _Emotion_Spotting_Service.model = model_from_json(loaded_model_json)

        # load weights to model
        _Emotion_Spotting_Service.model.load_weights(JSON_MODEL_WEIGHTS)

        # compile model
        _Emotion_Spotting_Service.model.compile(optimizer=keras.optimizers.Adam(), loss=losses.categorical_crossentropy, metrics=['acc'])

    return _Emotion_Spotting_Service._instance


if __name__ == '__main__':

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    ess = Emotion_Spotting_Service()

    gender1, emotion1 = ess.predict('/home/kavan/Documents/GA/Lessons/DSI-7-lessons_local/Capstone/capstone project_submission/test/03-01-03-02-01-02-04.wav')
    # gender2, emotion2 = ess.predict('../test/03-01-06-02-02-02-06.wav')
    # gender3, emotion3 = ess.predict('../test/03-01-08-02-02-01-02.wav')

    print(f"The audio clip uploaded is of a {gender1} \nWho feels {emotion1}!")

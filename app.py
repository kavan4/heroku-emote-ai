"""
server

client -> POST request -> server -> prediction back to client

"""
# imports
import random
from flask import Flask, request, jsonify, render_template
from emotion_spotting_service_boot import Emotion_Spotting_Service
import os
import tensorflow as tf

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024 # Max 3mb file can be uploaded

'''
say domain name is es.com/predict
whenever we send a request, flask gets a request and routes it to 
predict. 
'''


@app.route('/')
def home():
    return render_template('index_bootstrap.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    # allocate only the required memory to gpu
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    
    if request.method == "POST":
        # if no file uploaded
        if request.files['file'].filename == '':
            predicted_gender='None'
            predicted_emotion='None'
            return render_template('index_bootstrap.html',
                           prediction_text='The uploaded audio is of a {} who feels {}!'.format(predicted_gender, predicted_emotion))
        # make prediction
        elif request.files:
            # get audio file and save it
            audio_file = request.files["file"]
            file_name = str(random.randint(0, 100_000))
            audio_file.save(file_name)

            # invoke the spotting service
            ess = Emotion_Spotting_Service()

            # make a prediction
            predicted_gender, predicted_emotion = ess.predict(file_name)

            # remove the audio file
            os.remove(file_name)

    return render_template('index_bootstrap.html',
                           prediction_text='The uploaded audio is of a {} who feels {}!'.format(predicted_gender, predicted_emotion))

@app.route('/record', methods=['GET', 'POST'])
def record():

    return render_template('index_sr.html')

if __name__ == '__main__':
    app.run(debug=True)

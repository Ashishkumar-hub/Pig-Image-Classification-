# -*- coding: utf-8 -*-
"""
Created on: 5th May 2022
Company: DriFly Technologies Pvt. Ltd.
@author: Ashish Kumar
"""

# coding=utf-8
from __future__ import division, print_function
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import logging
import os

app = Flask(__name__)

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
logging.basicConfig(filename='test.log', filemode='w+', format='%(asctime)s %(message)s')
lg = logging.getLogger()
lg.addHandler(c_handler)
open(os.getcwd() + 'test.log', 'a')

# Model saved with Keras model.save()
MODEL_PATH = 'pig.h5'

# Loading trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img) # Preprocessing the image
        x=x/255 # Scaling # x = np.true_divide(x, 255)
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        preds=np.argmax(preds, axis=1)
        message = " Model Prediction Section"
        #print(preds)
        lg.info(str(preds) + " " + message)
        if preds==0:
            preds="This is a Landrace"
        else:
            preds="This is a Large White Yorkshire"
        return preds

    except Exception as e:
        lg.error(e)
        return render_template('error.html', message="Check logs for more info")

@app.route('/', methods=['GET'])
def index():
    try:
        return render_template('index.html') #main page
    except Exception as e:
        lg.error(e)
        return render_template('error.html', message="Check logs for more info")

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    try:
        if request.method == 'POST':
            f = request.files['file'] # Get the file from post request
            basepath = os.path.dirname(__file__) # Save the file to ./uploads
            file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            preds = model_predict(file_path, model) # Make prediction
            result=preds
            message = "prediction done"
            lg.info(str(preds) + " " + message)
            return result
        return None
    except Exception as e:
        lg.error(e)
        return render_template('error.html', message="Check logs for more info")

if __name__ == '__main__':
    app.run(debug=True)

# -*- coding: utf-8 -*-
"""
Created on Tue May 19 18:19:07 2020

@author: Pranita
"""

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from joblib import load
app = Flask(__name__)
model = load('dt.save')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[float(x) for x in request.form.values()]]
    print(x_test)
   
    
   
    
    prediction= model.predict(x_test)
    print(prediction)
    output=prediction[0]
    
    if(output==1):
        pred="NO ckd"
    else:
        pred="ckd"
    
    return render_template('index.html', prediction_text='Patient has {}'.format(pred))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)

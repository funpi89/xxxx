# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 09:49:41 2019

@author: a0922
"""

import pickle
from flask import Flask, request
import numpy as np
import pandas as pd
from flasgger import Swagger

with open('./rf.pkl','rb') as model_file:
    model = pickle.load(model_file)
    
app = Flask(__name__)
swagger = Swagger(app)

@app.route('/prediction')
def predic_iris():
    """Example endpoint returing a prediction of iris
    This is using docstrings for specifications.
    ---
    parameters:
        - name: s_length
          in: query
          type: number
          required: true
        - name: s_width
          in: query
          type: number
          required: true
        - name: p_length
          in: query
          type: number
          required: true
        - name: p_width
          in: query
          type: number
          required: true
    definitions:
        value:
            type: object
            properties:
                value_name:
                    type: string
                    items:
                        $ref: '#/definitions/Color'
        Color:
            type: string
    responses:
        200:
            description: OK
            schema:
                $ref: '#/definitions/value'
    """
    s_length = request.args.get("s_length")
    s_width = request.args.get("s_width")
    p_length = request.args.get("p_length")
    p_width = request.args.get("p_width")
    
    prediction = model.predict(np.array([[s_length, s_width, p_length, p_width]]))
    
    return str(prediction)

@app.route('/predict_file',methods=["POST"])
def predic_iris_file():
    """Example endpoint returing a prediction of iris
    This is using docstrings for specifications.
    ---
    parameters:
        - name: input_file
          in: formData
          type: file
          required: true
    definitions:
        value:
            type: object
            properties:
                value_name:
                    type: string
                    items:
                        $ref: '#/definitions/Color'
        Color:
            type: string
    responses:
        200:
            description: OK
            schema:
                $ref: '#/definitions/value'
    """
    input_data = pd.read_csv(request.files["input_file"], header=None)    
    prediction = model.predict(input_data)    
    return str(list(prediction))

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=5000,use_evalex=False,threaded=True)
    

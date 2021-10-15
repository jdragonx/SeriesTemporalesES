# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 23:21:07 2020

@author: jonhe
"""

import numpy as np
import pickle

from flask import Flask,request

scalerES = pickle.load(open( 'SeriesTemporalesES/scaler.p', 'rb' ))
modelES = pickle.load(open( 'SeriesTemporalesES/model.p', 'rb' ))

scalerEURUSD = pickle.load(open( 'SeriesTemporalesEURUSD/scaler.p', 'rb' ))
modelEURUSD = pickle.load(open( 'SeriesTemporalesEURUSD/model.p', 'rb' ))

app = Flask(__name__)

@app.route('/',methods=['GET'])
def prueba():
    return 'hola'

@app.route('/ES',methods=['POST'])
def prediccion():
    content = request.get_json()
    X = getX(content,20)
    Xt = scalerES.transform(X)
    Xt = Xt.flatten()
    Xt = np.atleast_2d(Xt)
    results = modelES.predict(Xt)
    results = results.reshape(len(Xt)*10,3)
    r_scale = scalerES.inverse_transform(results)
    r_scale = r_scale.flatten()
    count = 0
    for i in range(len(r_scale)):
        count += r_scale[i]
    return str(count)

@app.route('/EURUSD',methods=['POST'])
def prediccion():
    content = request.get_json()
    X = getX(content,30)
    Xt = scalerEURUSD.transform(X)
    Xt = Xt.flatten()
    Xt = np.atleast_2d(Xt)
    results = modelES.predict(Xt)
    results = results.reshape(len(Xt)*10,3)
    r_scale = scalerES.inverse_transform(results)
    r_scale = r_scale.flatten()
    count = 0
    for i in range(len(r_scale)):
        count += r_scale[i]
    return str(count)

def getX(content,n):
    X = [content['bar'+str(1)]]
    for i in range(1,n):
        X = np.concatenate((X,[content['bar'+str(i+1)]]))
    return X


if __name__ == '__main__':
    app.run()
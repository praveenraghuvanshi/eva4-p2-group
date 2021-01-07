import os
import io
import json
import re
from flask import Flask
from flask import request, jsonify, make_response
from train import train_model, predict_sentiment, upload_data
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, World!"

@app.route("/upload", methods=['POST'])
def upload():
    print("Inside upload")
    f:any = None
    uploadedFileUrl = ''
    if request.method == 'POST':
        print(request.files)
        f = request.files['']
        f.save(secure_filename(f.filename))
        uploadedFileUrl = upload_data(f.filename)     

    return {
        "file" : f.filename,
        "uploadedFileUrl" : uploadedFileUrl,
    } 

@app.route("/train")
def train():
    data_file = request.args.get('data')
    print(data_file)
    return train_model(data_file)

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == "OPTIONS": # CORS preflight
        return _build_cors_prelight_response()
    elif request.method == "POST": # The actual request following the preflight
        sentence  = request.json['inputtext']
        modelName = request.json['model']
        textFields = request.json["textfields"]
        
        predValue = predict_sentiment(sentence, modelName, textFields)
        print('Predicted value',predValue)
        review = "Positive" if predValue >= 0.5 else "Negative"
        print(review)
        response = make_response()
        return _corsify_actual_response(jsonify({'prediction': review}))
    else:
        raise RuntimeError("Weird - don't know how to handle method {}".format(request.method))

def _build_cors_prelight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
    # app.run(debug=True)
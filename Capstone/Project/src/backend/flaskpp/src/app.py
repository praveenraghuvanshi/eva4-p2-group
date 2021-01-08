import os
import json
from flask import Flask
from flask import request, jsonify, make_response
from flask_cors import CORS

from uploaddownload import upload_file
from sentimentanalysis import train_model_sa, predict_sentiment

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Hello, World!"

@app.route("/upload", methods=['POST'])
def upload():
    print("Upload started")
    if request.method == "OPTIONS": # CORS preflight
        return _build_cors_prelight_response()
    elif request.method == "POST": # The actual request following the preflight
        f:any = None
        uploadedFileUrl = ''
        print(request.files)
        f = request.files['file']
        uploadedFileUrl = upload_file(f)

        response = {
                "file" : f.filename,
                "uploadedFileUrl" : uploadedFileUrl,
            }
        print(f'Upload completed: {response}')
        return _corsify_actual_response(jsonify(response))

@app.route("/train/sa")
def train_sa():
    data_file = request.args.get('data')
    print(data_file)
    return train_model_sa(data_file)

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == "OPTIONS": # CORS preflight
        return _build_cors_prelight_response()
    elif request.method == "POST": # The actual request following the preflight
        print('Inside POST')
        print(request.data)
        sentence  = json.loads(request.data)['inputtext']
        modelName = json.loads(request.data)['model']
        textFields = json.loads(request.data)["textfields"]
        
        predValue = predict_sentiment(sentence, modelName, textFields)
        print('Predicted value',predValue)
        review = "Positive" if predValue >= 0.5 else "Negative"
        print(review)
        response = make_response()
        return _corsify_actual_response(jsonify({'prediction': review}))
    else:
        raise RuntimeError("Weird - don't know how to handle method {}".format(request.method))

'''Handling CORS issues'''
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

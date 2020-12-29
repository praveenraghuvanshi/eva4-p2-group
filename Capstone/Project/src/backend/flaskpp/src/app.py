import os
import io
import json
import re
from flask import Flask
from flask import request
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
    sentence  = request.json['inputtext']
    modelName = request.json['model']
    textFields = request.json["textfields"]
    textVocab = request.json["textvocab"]

    predValue = predict_sentiment(sentence, modelName, textFields, textVocab)
    print('Predicted value',predValue)
    review = "Positive" if predValue >= 0.5 else "Negative"
    print(review)
    return review

if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=80, debug=True)
    app.run(debug=True)
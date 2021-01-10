import json
import base64
from flask import Flask
from flask import request, jsonify, make_response
from flask_cors import CORS

import sentimentanalysis
import imageclassification
from uploaddownload import upload_file

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Hello, World!"

@app.route("/upload", methods=['POST'])
def upload():
    # print("Upload started")
    try:
        if request.method == "OPTIONS": # CORS preflight
            return _build_cors_prelight_response()
        elif request.method == "POST": # The actual request following the preflight
            f:any = None
            uploadedFileUrl = ''
            # print(request.files)
            f = request.files['file']
            uploadedFileUrl = upload_file(f)

            response = {
                        "file" : f.filename,
                        "uploadedFileUrl" : uploadedFileUrl,
                    }
            # print(f'Upload completed: {response}')
            return _corsify_actual_response(jsonify(response))
    except Exception as e:
        print(str(e))
        return _corsify_actual_response(jsonify({'result': ""}))

@app.route("/train/sa")
def train_sa():
    data_file = request.args.get('data')
    print(data_file)
    return sentimentanalysis.train_model(data_file)

@app.route("/train/ic")
def train_ic():
    data_file = request.args.get('data')
    print(data_file)
    (model, test_loss, test_acc, class_to_idx) = imageclassification.train_model(data_file)
    return {
        "model": model,
        "test_loss" : test_loss,
        "test_acc": test_acc,
        "class_to_idx": class_to_idx
    }

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
        
        predValue = sentimentanalysis.predict_sentiment(sentence, modelName, textFields)
        print('Predicted value',predValue)
        review = "Positive" if predValue >= 0.5 else "Negative"
        print(review)
        return _corsify_actual_response(jsonify({'prediction': review}))
    else:
        raise RuntimeError("Weird - don't know how to handle method {}".format(request.method))

@app.route("/classify", methods=['POST'])
def classify():
    if request.method == "OPTIONS": # CORS preflight
        return _build_cors_prelight_response()
    elif request.method == "POST": # The actual request following the preflight
        print('Inside POST')

        img_data = json.loads(request.data)['image']
        modelName = json.loads(request.data)['model']

        img_data = img_data[23:]
        encoded=img_data.encode('utf-8')
        array=bytearray(encoded)
        imagePath = "ImageToBePredicted.png"
        with open(imagePath, "wb") as fh:
            fh.write(base64.decodebytes(array))

        result = imageclassification.classify_image(imagePath, modelName)
        return _corsify_actual_response(jsonify({'result': result}))
    else:
        raise RuntimeError("Weird - don't know how to handle method {}".format(request.method))

@app.route('/clear',methods =['GET'])
def clearDataset():
    data_file = request.args.get('directory')
    print(data_file)
    try:
        import shutil 
        shutil.rmtree(data_file)
        print('\n\n\n*********** Deleted the dataset Directory ***********')
    except Exception as e:
        print(str(e))
        #return {"result" : False}
    return {"result" : True}

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

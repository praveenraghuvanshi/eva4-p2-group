from flask import Flask
from flask import request
from train import train_model, predict_sentiment

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, World!"

@app.route("/train")
def train():
    data_file = request.args.get('data')
    print(data_file)
    return train_model(data_file)

@app.route("/predict", methods=['POST'])
def predict():
    input  = request.json['inputtext']
    modelName = request.json['model']

    predValue = predict_sentiment(modelName,input)
    print('Predicted value',predValue)
    review = "Positive" if predValue >= 0.5 else "Negative"
    print(review)
    return review

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
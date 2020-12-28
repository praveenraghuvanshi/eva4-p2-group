from flask import Flask
from train import train_model

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, World!"

@app.route("/train")
def train():
    return train_model()

if __name__ == "__main__":
    app.run(debug=True)
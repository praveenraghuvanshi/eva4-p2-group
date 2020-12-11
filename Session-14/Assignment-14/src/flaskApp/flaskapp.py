import os
import requests
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from pprint import pprint

# URLS/API
image_classify_mobilenet_url = 'https://daa70df9c6.execute-api.ap-south-1.amazonaws.com/dev/classify'

app = Flask(__name__)
app.debug = True

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/uploader', methods = ['POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))

      url = image_classify_mobilenet_url
      files = {'file': open(f.filename, 'rb')}
      response = requests.post(
          url,
          files = files
      )

      print('Status code', response.status_code)
      print('Content', response.text)

      pprint(response.json())

      if response.ok:
        return response.text
      else:
        return f"Error uploading file! - {response.status_code}, {response.text}"

if __name__ == '__main__':
    app.run()
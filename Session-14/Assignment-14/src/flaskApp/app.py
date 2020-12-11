import os
import requests
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# URLS/API


app = Flask(__name__)
app.debug = True
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'uploads'

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/uploader', methods = ['POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))

      url = 'https://41qjwezhob.execute-api.ap-south-1.amazonaws.com/dev/classify'
      files = {'media': open(f.filename, 'rb')}
      response = requests.post(
          url,
          files = files
      )

      print('Status code', response.status_code)
      print('Content', response.text)

      if response.ok:
        return "File uploaded!"
      else:
        return f"Error uploading file! - {response.status_code}, {response.text}"

if __name__ == '__main__':
    app.run()
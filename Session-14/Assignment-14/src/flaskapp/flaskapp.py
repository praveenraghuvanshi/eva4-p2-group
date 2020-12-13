try:
    import os,io
    import requests
    from flask import Flask, render_template, request
    from werkzeug.utils import secure_filename
    from pprint import pprint
except Exception as e:
    print(str(e))
    

# Handler file code:
try:
    import json
    import sys
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from PIL import Image

    from torchvision.models import resnet50
    from torch import nn
except Exception as e:
    print(str(e))

    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'### device : {device}')
model = torch.jit.load('model.pt', map_location=torch.device('cpu'))
model.to(device)
print('### Model Mobilenet is :{}'.format(model))
    
    
def transform_image(image_bytes):
    print('Transformation Start...')
    try:
        transformations = transforms.Compose([
            transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = Image.open(io.BytesIO(image_bytes))
        print('Transformation finished...')
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print('Exception in transforming an image')
        print(repr(e))
        raise(e)
        
        
def get_prediction(image_bytes):
    print('Getting Prediction...')
    tensor = transform_image(image_bytes=image_bytes)
    return model(tensor).argmax().item()
    
    
    

app = Flask(__name__, template_folder='templates')
app.debug = True

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/uploader', methods = ['POST'])
def upload_file():
    print('### U r in uploadfile func')
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        from PIL import Image
        im = Image.open(f.filename)
        img_byte_arr = io.BytesIO()
        im.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        prediction = get_prediction(image_bytes=img_byte_arr)
        print('### prediction : {}'.format(prediction))
        
        classes = {0: 'Flying Bird', 1 : 'Large Quadcopter', 2 : 'Small Quadcopter', 3 : 'Winged Drone'}
        
        return classes[prediction]
        

if __name__ == '__main__':
    app.run(debug=True)
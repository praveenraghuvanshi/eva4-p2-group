print("Import START...")
try:
    import unzip_requirements
except ImportError:
    pass
from requests_toolbelt.multipart import decoder
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn

import boto3
import sys

import os
import io
import base64
import json

#import dlib
import numpy as np

print("Import END...")

# define env variables if there are not existing
S3_BUCKET = os.environ['MODEL_BUCKET_NAME'] if 'MODEL_BUCKET_NAME' in os.environ else 'suman-p2-bucket'
MODEL_PATH = os.environ['MODEL_FILE_NAME_KEY'] if 'MODEL_FILE_NAME_KEY' in os.environ else 'detectFacesUpd.pt'
#MODEL_PATH = os.environ['MODEL_FILE_NAME_KEY'] if 'MODEL_FILE_NAME_KEY' in os.environ else 'detectMyFace_suman_1.5.pt'
#MODEL_PATH = os.environ['MODEL_FILE_NAME_KEY'] if 'MODEL_FILE_NAME_KEY' in os.environ else 'model.pt'

print(S3_BUCKET)
print(MODEL_PATH)

print('Downloading model...')


# load the S3 client when lambda execution context is created
s3 = boto3.client('s3')
# print('s3 called')

def load_model_from_s3():
    try:
        cuda = torch.cuda.is_available()
        print("CUDA Available?", cuda)
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print('Creating Bytestream')
        bytestream = io.BytesIO(obj['Body'].read())
        print('Loading model...')
        print(sys.getsizeof(bytestream) // (1024 * 1024))
        model = torch.jit.load(bytestream, map_location=torch.device('cpu'))
        print(model is None)
        return model
    except Exception as e:
        print('Exception in loading a model')
        print(repr(e))
        raise(e)

model = load_model_from_s3()
print(model is None)


def face_landmark_detector():
    faceDetector = dlib.get_frontal_face_detector()
    landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)
    return (faceDetector, landmarkDetector)

       
def transform_image(image_bytes):
    print('Transformation Start...')
    try:
        transformations = transforms.Compose([
                                transforms.ToTensor()])
        image = Image.open(io.BytesIO(image_bytes))
        print('Transformation finished...')
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print('Exception in transforming an image')
        print(repr(e))
        raise(e)


def img_to_base64(img):
    img = Image.fromarray(img, 'RGB') 
    buffer = io.BytesIO()
    img.save(buffer,format="JPEG")
    myimage = buffer.getvalue()                     
    img_str = f"data:image/jpeg;base64,{base64.b64encode(myimage).decode()}"
    return img_str

def get_aligned_image(image_bytes):
    print('Alignment process started...')
    (faceDetector, landmarkDetector) = face_landmark_detector()
    im = transform_image(image_bytes)

    # Detect landmarks
    points = fbc.getLandmarks(faceDetector, landmarkDetector, im)
    points = np.array(points)
    landmarksDetected = len(points)
    print(f'Landmarks detected: {landmarksDetected}')

    if landmarksDetected == 0:
        return None

    im = np.float32(im)/255.0

    h = im.shape[0] # 600
    w = im.shape[1] # 600

    imNorm, points = fbc.normalizeImagesAndLandmarks((h, w), im, points)
    imNorm = np.uint8(imNorm * 255)

    return imNorm # aligned image

def faceRecognition(image_bytes):
    print('### You are going to recognise face')
    tensor = transform_image(image_bytes=image_bytes)
    print(model is None)
    return model(tensor).argmax().item()
   

def classify_image(event, context):
    try:
        #classes = ['dhanu', 'dhoni', 'suman', 'sushant']
        content_type_header = event['headers']['content-type']
        body = base64.b64decode(event["body"])
        print('BODY Loaded')

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        prediction = faceRecognition(image_bytes=picture.content)
        print(prediction)
        #label = classes[prediction]

        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]
            
        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({'file': filename.replace('"', ''), 'predicted': prediction})
        }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }

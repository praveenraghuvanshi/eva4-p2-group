print("Import START...")
try:
    import unzip_requirements
except ImportError:
    pass
from requests_toolbelt.multipart import decoder
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from torchvision.models import resnet50
from torch import nn

import boto3
import os
import io
import base64
import json
import sys
from requests_toolbelt.multipart import decoder

import dlib
import cv2
import numpy as np
import faceBlendCommon as fbc

print("Import END...")

# print(os.getcwd())

# define env variables if there are not existing
S3_BUCKET = os.environ['MODEL_BUCKET_NAME'] if 'MODEL_BUCKET_NAME' in os.environ else 'eva4p2bucket1'
MODEL_PATH = os.environ['MODEL_FILE_NAME_KEY'] if 'MODEL_FILE_NAME_KEY' in os.environ else 'model.pt'
PREDICTOR_PATH = os.environ['PREDICTOR_5_FACE_DETECTOR_KEY'] if 'PREDICTOR_5_FACE_DETECTOR_KEY' in os.environ else 'shape_predictor_5_face_landmarks.dat'

# load the S3 client when lambda execution context is created
s3 = boto3.client('s3')

def load_model_from_s3():
    try:
        cuda = torch.cuda.is_available()
        print("CUDA Available?", cuda)

        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print('Creating Bytestream')
        bytestream = io.BytesIO(obj['Body'].read())
        print('Loading model...')
        print(f'Model Size: {sys.getsizeof(bytestream) // (1024 * 1024)}')
        model = torch.jit.load(bytestream, map_location=torch.device('cpu'))
        print(model is None)
        return model
    except Exception as e:
        print('Exception in loading a model')
        print(repr(e))
        raise(e)

model = load_model_from_s3()
print(model is None)

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
    print('Getting Prediction - bytes...')
    tensor = transform_image(image_bytes=image_bytes)
    return model(tensor).argmax().item()

def get_prediction(image_tensor):
    print('Getting Prediction - tensor...')
    return model(image_tensor).argmax().item()

def get_aligned_image(image_bytes):
    print('Image contains face')
    # 1. Identify if face is there or not
    # 2. Return aligned image

    faceDetector = dlib.get_frontal_face_detector()
    landmarkDetector = dlib.shape_predictor("s3://eva4p2bucket1/shape_predictor_5_face_landmarks.dat")

    # imageFileName = DATA_PATH + "images/face.jpg"
    # im = cv2.imread(imageFileName)

    im_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    im = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

    points = fbc.getLandmarks(faceDetector, landmarkDetector, im)
    points = np.array(points)
    print(len(points))

    im = np.float32(im)/255.0

    h = 600
    w = 600

    imNorm, points = fbc.normalizeImagesAndLandmarks((h, w), im, points)
    imNorm = np.uint8(imNorm * 255)

    return imNorm

def classify_image(event, context):
    try:
        content_type_header = event['headers']['content-type']
        body = base64.b64decode(event["body"])
        print('BODY Loaded')

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        # image_tensor = transform_image(image_bytes=picture.content)
        imNorm = get_aligned_image(picture.content)
        print(type(imNorm))
        # prediction = get_prediction(image_tensor=image_tensor)
        # print(f'Predicted value: {prediction}')

        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]

        image = Image.open(io.BytesIO(picture.content))

        '''body = base64.b64decode(event["body"])
        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]        
        print(f'type of image: {type(picture.content)}')
        print(picture.content)
        encoded_img = base64.b64encode(picture.content).decode('ascii') # picture.content.decode('utf-8').replace("'", '"')
        print(f'Response image: {type(encoded_img)}')'''

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({'file': filename.replace('"', ''), 'predicted': prediction, 'containsface': str(imageHasFace), 'imagebytes': imNorm})
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
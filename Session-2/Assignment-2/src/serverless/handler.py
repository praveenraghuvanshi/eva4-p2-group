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
print("Import END...")

# print(os.getcwd())

# define env variables if there are not existing
S3_BUCKET = os.environ['MODEL_BUCKET_NAME'] if 'MODEL_BUCKET_NAME' in os.environ else 'eva4p2bucket1'
MODEL_PATH = os.environ['MODEL_FILE_NAME_KEY'] if 'MODEL_FILE_NAME_KEY' in os.environ else 'model.pt'

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
    print('Getting Prediction...')
    tensor = transform_image(image_bytes=image_bytes)
    return model(tensor).argmax().item()


def classify_image(event, context):
    try:
        print(json.dumps(event))
        content_type_header = event['headers']['Content-Type']
        body = base64.b64decode(event["body"])
        print('BODY Loaded')

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        prediction = get_prediction(image_bytes=picture.content)
        print(f'Predicted value: {prediction}')

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
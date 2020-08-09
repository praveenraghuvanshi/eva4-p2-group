# EVA-4 Phase-2 : Assignment - 2

### Team Members

- Praveen Raghuvanshi (praveenraghuvanshi@gmail.com)
- Tusharkanta Biswal (Tusharkanta_biswal@stragure.com)
- Suman Kanukollu (sumankanukollu@gmail.com)
- Shubham Kothawade (kothawadeshub@gmail.com)

## Resources

- Github: https://github.com/praveenraghuvanshi/eva4-p2-group/tree/master/Session-2/Assignment-2
- Source: https://github.com/praveenraghuvanshi/eva4-p2-group/tree/master/Session-2/Assignment-2/src
- Notebook: https://github.com/praveenraghuvanshi/eva4-p2-group/blob/master/Session-2/Assignment-2/src/eva4p2s2.ipynb
- Rest API: https://41qjwezhob.execute-api.ap-south-1.amazonaws.com/dev/classify

## Result

**Input Image**

**Flying Bird**

<img src=".\assets\flyingbird.png" alt="Small Quadroptor" style="zoom:80%;" />

Predicted: 

```javascript
{
  "file": "flyingbird.png",
  "predicted": 0
}
```



**Large Qaudraptor**

<img src=".\assets\largeQ.png" alt="Small Quadroptor" style="zoom:50%;" />

Predicted: 

```javascript
{
  "file": "largeQ.png",
  "predicted": 1
}
```



**Small Quadraptor**

<img src=".\assets\smallQ.png" alt="Small Quadroptor" style="zoom:80%;" />



Predicted:

```javascript
{
  "file": "smallQ.png",
  "predicted": 2
}
```

**Winged Drone**

<img src=".\assets\wingeddrone.png" alt="Small Quadroptor" style="zoom:80%;" />



Predicted:

```javascript
{
  "file": "wingeddrone.png",
  "predicted": 3
}
```



**REST Client**

<img src=".\assets\rest-prediction.png" alt="Rest Client - Prediction" style="zoom:80%;" />



**Cloudwatch Logs**

<img src=".\assets\cloudwatch-logs.png" alt="cloudwatch-logs" style="zoom:80%;" />

## Source

**handle.py**

```python
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
MODEL_PATH = os.environ['MODEL_FILE_NAME_KEY'] if 'MODEL_FILE_NAME_KEY' in os.environ else 'modelnet_v2_44_77.pt'

'''if 'MODEL_BUCKET_NAME' in os.environ:
    print(os.environ['MODEL_BUCKET_NAME'])

if 'MODEL_FILE_NAME_KEY' in os.environ:
    print(os.environ['MODEL_FILE_NAME_KEY'])'''

print(S3_BUCKET)
print(MODEL_PATH)

print('Downloading model...')

# load the S3 client when lambda execution context is created
s3 = boto3.client('s3')
# print('s3 called')

def load_model_from_s3():
    try:
        # print('Inside try...')
        # print(MODEL_PATH)
        # print(os.path.isfile(MODEL_PATH))
        # if os.path.isfile(MODEL_PATH) != True:
        # print('Inside if...')
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
    # print(model is None)
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
    # print(model is None)
    tensor = transform_image(image_bytes=image_bytes)
    print(model is None)
    return model(tensor).argmax().item()


def classify_image(event, context):
    # print(model is None)
    try:
        content_type_header = event['headers']['content-type']
        # print(event['body'])
        # print(event['headers'])
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
```

**serverless.yml**

```python
# Welcome to Serverless!
#
# This file is the main config file for your service.
# It's very minimal at this point and uses default values.
# You can always add more config options for more control.
# We've included some commented out config examples here.
# Just uncomment any of them to get that config option.
#
# For full config options, check the docs:
#    docs.serverless.com
#
# Happy Coding!

service: eva4-p2-assignment-1
# app and org for use with dashboard.serverless.com
#app: your-app-name
#org: your-org-name

# You can pin your service to only deploy with a specific Serverless version
# Check out our docs for more details
# frameworkVersion: "=X.X.X"

provider:
  name: aws
  runtime: python3.8
  stage: dev
  region: ap-south-1
  timeout: 60

  environment:
    MODEL_BUCKET_NAME: eva4p2bucket1
    MODEL_FILE_NAME_KEY: mobilenet_v2_44_77.pt
  iamRoleStatements:
    - Effect: "Allow"
      Action:
        - s3:getObject
      Resource: arn:aws:s3:::eva4p2bucket1/*

custom:
  pythonRequirements:
    dockerizePip: true
    zip: true
    slim: true
    strip: false
    noDeploy:
      - docutils
      - jmespath
      - pip
      - python-dateutil
      - setuptools
      - six
      - tensorboard
    useStaticCache: true
    useDonwloadCache: true
    cacheLocation: "./cache"

package:
  individually: false
  exclude:
    - package.json
    - package-log.json
    - node_modules/**
    - cache/**
    - test/**
    - __pycache__/**
    - .pytest_cache/**
    - model/**

functions:
  classify_image:
    handler: handler.classify_image
    memorySize: 3008
    timeout: 60
    events:
      - http:
          path: classify
          method: post
          cors: true

plugins:
  - serverless-python-requirements
```


print("Import START...")
try:
    import unzip_requirements
except ImportError:
    pass
from requests_toolbelt.multipart import decoder
from PIL import Image

import os
import io
import base64
import json

import dlib
import numpy as np
import faceBlendCommon as fbc

print("Import END...")

# print(os.getcwd())

# define env variables if there are not existing
# PREDICTOR_PATH = os.environ['PREDICTOR_5_FACE_DETECTOR_KEY'] if 'PREDICTOR_5_FACE_DETECTOR_KEY' in os.environ else 'shape_predictor_5_face_landmarks.dat'
PREDICTOR_PATH = 'shape_predictor_5_face_landmarks.dat'

faceDetector = dlib.get_frontal_face_detector()
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

def transform_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return np.array(image)
    except Exception as e:
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

    im = transform_image(image_bytes)

    # Detect landmarks
    points = fbc.getLandmarks(faceDetector, landmarkDetector, im)
    points = np.array(points)
    print(len(points))

    im = np.float32(im)/255.0

    h = im.shape[0] # 600
    w = im.shape[1] # 600

    imNorm, points = fbc.normalizeImagesAndLandmarks((h, w), im, points)
    imNorm = np.uint8(imNorm * 255)

    return imNorm # aligned image

def classify_image(event, context):
    try:
        content_type_header = event['headers']['content-type']
        body = base64.b64decode(event["body"])
        print('BODY Loaded')

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        # image_tensor = transform_image(image_bytes=picture.content)
        aligned_image = get_aligned_image(picture.content)
        print('Alignment Done!!!')
        print(type(aligned_image))
        # prediction = get_prediction(image_tensor=image_tensor)
        # print(f'Predicted value: {prediction}')

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
            "body": json.dumps({'file': filename.replace('"', ''), 'imagebytes':  img_to_base64(aligned_image)})
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
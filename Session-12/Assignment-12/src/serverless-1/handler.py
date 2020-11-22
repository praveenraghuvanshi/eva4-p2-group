try:
    print('### Import started')
    import os
    import io
    import base64
    import json
    import sys
    import unzip_requirements
    from requests_toolbelt.multipart import decoder
    print('### Imported unzip_requirements')

    import torch
    import torch.nn.functional as F
    print('### Import torch completed')

    import torchvision.transforms as transforms    
    print('### Import torchvision completed')

    print('### Import End....')
except ImportError:
    print('### Exception occurred in import')

def generatecaption(event, context):
    try:
        print('### You are in generatecaption method')
        print(f'### Event is : {event}')
        content_type_header = event['headers']['content-type']
        body = base64.b64decode(event["body"])
        print('BODY Loaded')

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        caption = "Image response" # generateCaptionInternal(picture.Content)

        response = {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({"output":caption})
        }

        return response
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

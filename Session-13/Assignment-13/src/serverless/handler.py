try:
    print('### Import started')
    import boto3
    import os
    import io
    import base64
    import json
    import sys
    from requests_toolbelt.multipart import decoder

    import numpy as np

    print('### Import basic completed')
    import torch
    import torchaudio
    
    print('### Import torch completed')
  
    print('### Import End....')
except ImportError:
    print('### Exception occurred in import')



def hello(event, context):
    body = {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "input": event
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response

    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
    """
    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }
    """

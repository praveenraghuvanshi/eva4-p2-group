print("Import START...")
try:
    import unzip_requirements
except ImportError:
    pass

from requests_toolbelt.multipart import decoder
import json
import base64
import os
import urllib
import datetime
import boto3
from botocore.client import Config

BUCKET_NAME = os.environ['BUCKET_NAME'] if 'BUCKET_NAME' in os.environ else 'aiendeavour'
print(BUCKET_NAME)

s3 = boto3.resource(u's3')

def upload(event, context):
    try:
        content_type_header = event['headers']['Content-Type']        
        body = base64.b64decode(event["body"])
        print(type(body))
        print('BODY Loaded')

        csv = decoder.MultipartDecoder(body, content_type_header).parts[0]

        # Upload to S3
        filename = csv.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = csv.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]

        filename = filename.replace('"', "")
        print(filename)

        """Make a variable containing the date format based on YYYYYMMDD"""
        cur_dt = datetime.datetime.today().strftime('%Y%m%d%H%M%S')

        # construct file name
        FILE_NAME = cur_dt + '_' + filename
        LOCAL_FILE_PATH = '/tmp/' + FILE_NAME

        # Save to temp location
        data = open(LOCAL_FILE_PATH, 'wb')
        data.write(csv.content)
        data.close

        # save to S3
        s3.Bucket(BUCKET_NAME).upload_file(LOCAL_FILE_PATH, FILE_NAME)
        uploadedFileUrl = "https://s3-%s.amazonaws.com/%s/%s" % (
            "ap-south-1",
            BUCKET_NAME,
            urllib.parse.quote(FILE_NAME, safe="~()*!.'"),
        )
        print('S3 uploaded file Url is ' + FILE_NAME)
        print("S3 url is " + uploadedFileUrl)

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({ 'uploadedfile': FILE_NAME, 'resourceurl':  uploadedFileUrl})
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
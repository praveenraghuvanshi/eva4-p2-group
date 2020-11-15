import logging

import azure.functions as func
import os, uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

import io, sys, pickle
import torch, torchtext
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

def load_text_vocab(fileName):
    try:
        print("Azure Blob storage v" + __version__ + " - Python quickstart sample")
        logging.info("Azure Blob storage v" + __version__ + " - Python quickstart sample")
        # Quick start code goes here
        # connection_string = "DefaultEndpointsProtocol=https;AccountName=slswusdevd70d42;AccountKey=hfN/JLO4ZU6ivVg4EpxGaBrYWPOuJxJAM97m7lmn3NzVG2nrhBOZGe+T/Lthl/SDQFHreNoaewTliPEE/6lk7A==;EndpointSuffix=core.windows.net"
        # connection_string = "UseDevelopmentStorage=true"
        connection_string = "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;"
        service = BlobServiceClient.from_connection_string(conn_str=connection_string)

        blob = BlobClient.from_connection_string(conn_str=connection_string, container_name="sentimentstore", blob_name=fileName)
        #get the length of the blob file, you can use it if you need a loop in your code to read a blob file.
        blob_property = blob.get_blob_properties()

        print("the length of the blob is: " + str(blob_property.size // (1024 * 1024)) + " MB")
        print("**********")
        logging.info("the length of the blob is: " + str(blob_property.size // (1024 * 1024)) + " MB")
        logging.info("**********")
        
        stream = blob.download_blob()
        blob_data = stream.content_as_bytes()
        print(type(blob_data))
        logging.info(type(blob_data))
        bytestream  = io.BytesIO(blob_data)
        print(type(bytestream))
        logging.info(type(bytestream))

        print(f'### Model Text Vocab obj Size: {sys.getsizeof(bytestream) // (1024 * 1024)}')
        logging.info(f'### Model Text Vocab obj Size: {sys.getsizeof(bytestream) // (1024 * 1024)}')
        with bytestream as f: 
          txt_vocab = pickle.load(f)
        # txt_vocab = pickle.load(bytestream)
        print('### type of text vocab : {}'.format(type(txt_vocab))) 
        logging.info('### type of text vocab : {}'.format(type(txt_vocab))) 
        INPUT_DIM = len(txt_vocab) #len(TEXT.vocab)
        print(f'### Input dimension: {INPUT_DIM}')
        logging.info(f'### Input dimension: {INPUT_DIM}')

    except Exception as ex:
        print('Exception:')
        print(ex)
        logging.info('Exception')
        logging.info(ex)

load_text_vocab("s9_wordemb_text_vocab_1.pkl")

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a sentiment request.')

    try:
        req_body = req.get_json()
        message = req_body.get('message')
        print(message)        
    except ValueError:
        pass

    if message:
        return func.HttpResponse(f'Message Received: {message}')
    else:
        return func.HttpResponse(
             "Please pass text for analysis",
             status_code=400
        )

try:
    print("### Import START...")
    import unzip_requirements
    import base64,boto3,os,io,json,sys

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchtext import data
    import random,pickle
    import numpy as np

    from model import *

    from io import BytesIO

    import spacy
    import dill
    print('### Using Torch version :',torch.__version__)
except Exception as e:
    print('### Exception occurred while importing modules : {}'.format(str(e)))

# define env variables if there are not existing
S3_BUCKET   = os.environ['MODEL_BUCKET_NAME'] if 'MODEL_BUCKET_NAME' in os.environ else 'eva4p2bucket1'
MODEL_PATH  = os.environ['MODEL_FILE_NAME_KEY'] if 'MODEL_FILE_NAME_KEY' in os.environ else 's11-tranlation.pt'
SRC  = os.environ['SRC'] if 'SRC' in os.environ else 'SRC.pkl'
TRG  = os.environ['TRG'] if 'TRG' in os.environ else 'TRG.pkl'
print('### S3 Bkt is : {} \nModel path is : {}'.format(S3_BUCKET,MODEL_PATH))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'### Device is : {device}')

s3 = boto3.client('s3')

def load_model_from_s3bkt():
    try:
        obj         = s3.get_object(Bucket = S3_BUCKET, Key = MODEL_PATH)
        bytestream  = io.BytesIO(obj['Body'].read())
        print('### Loading model...')
        print(f'Model Size: {sys.getsizeof(bytestream) // (1024 * 1024)}')
        model = torch.load(bytestream,map_location=torch.device('cpu'))
        print('### Model in eval mode : {}'.format(model.eval()))        
        print('### Model is loaded and returning model')
        return model
    except Exception as e:
        print('### Exception in loading a model : {}'.format(str(e)))
        raise(e)

def load_src_trg():
    try:
        obj         = s3.get_object(Bucket = S3_BUCKET, Key = SRC)
        bytestream  = io.BytesIO(obj['Body'].read())
        print('### Loading model...')
        print(f'Model Size: {sys.getsizeof(bytestream) // (1024 * 1024)}')
        SRC = torch.load(bytestream, pickle_module=dill)       
        print('### SRC is loaded')

        obj         = s3.get_object(Bucket = S3_BUCKET, Key = TRG)
        bytestream  = io.BytesIO(obj['Body'].read())
        print('### Loading model...')
        print(f'Model Size: {sys.getsizeof(bytestream) // (1024 * 1024)}')
        TRG = torch.load(bytestream, pickle_module=dill)       
        print('### TRG is loaded')

        return SRC, TRG
    except Exception as e:
        print('### Exception in loading a model : {}'.format(str(e)))
        raise(e)

model   = load_model_from_s3bkt()
(SRC, TRG) = load_src_trg()

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"    
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

device = 'cpu'
DEVICE = 'cpu'
print('Import End....')

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

def greedy_decode(model, src, src_mask, src_lengths, max_len=100, sos_index=1, eos_index=None):
    """Greedily decode a sentence."""

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)

    output = []
    attention_scores = []
    hidden = None

    for i in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output = model.decode(
              encoder_hidden, encoder_final, src_mask,
              prev_y, trg_mask, hidden)

            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data.item()
        output.append(next_word)
        prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
        attention_scores.append(model.decoder.attention.alphas.cpu().numpy())
    
    output = np.array(output)
        
    # cut off everything starting from </s> 
    # (only when eos_index provided)
    if eos_index is not None:
        first_eos = np.where(output==eos_index)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]      
    
    return output, np.concatenate(attention_scores, axis=1)

def lookup_words(x, vocab=None):
    if vocab is not None:
        x = [vocab.itos[i] for i in x]
            
    return [str(t) for t in x]

def translate(model, sentence,spacy_model,src_fields,DEVICE='cpu'):
    model.eval()
    tokenized = [tok.text for tok in spacy_model.tokenizer(sentence)]
    print(tokenized)
    indexed = [src_fields.vocab.stoi[t] for t in tokenized]
    print(indexed)
    tensor = torch.LongTensor(indexed).to(DEVICE)
    tensor = tensor.unsqueeze(0)
    print(tensor)
    print(tensor.size())
    pad_index = 0
    src_mask = (tensor != pad_index).unsqueeze(0).to(DEVICE)
    print(src_mask)
    print(src_mask.size())
    src_lengths = torch.LongTensor([tensor.size()[1]]).to(DEVICE)
    print(src_lengths)
    print(src_lengths.size())
    result, _ = greedy_decode(
          model, tensor, src_mask, src_lengths,
          max_len=25, sos_index=TRG.vocab.stoi[SOS_TOKEN], eos_index=TRG.vocab.stoi[EOS_TOKEN])
    
    output = lookup_words(result,vocab= TRG.vocab)
    outText = ' '.join(output)
        
    return outText

def translator(event, context):
    try:
        print('### You are in handler dcGAN_car function')
        print('### event is : {}'.format(event))
        print('### Context is : {}'.format(context))
        
        body = json.loads(event['body'])
        input = body["input"]
        print(input)

        result = translate(model,input,spacy_de,SRC,device)
        response = {
            "statusCode": 200,
            "body": json.dumps({"input": input , "output":result})
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
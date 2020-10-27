def emptyDir(folder):
    isdir = os.path.isdir(folder)  
    print(isdir)
    if isdir == False:
        return
 
    fileList = os.listdir(folder)
    print(len(fileList))
    for f in fileList:
        filePath = folder + '/'+f 
        if os.path.isfile(filePath):
            os.remove(filePath)
        elif os.path.isdir(filePath):
            newFileList = os.listdir(filePath)
            for f1 in newFileList:
                insideFilePath = filePath + '/' + f1
                if os.path.isfile(insideFilePath):
                    os.remove(insideFilePath)

try:
    print("### Import START...")
    import unzip_requirements
    import base64,boto3,os,io,json,sys

    # print('### Empty Directory')
    # emptyDir('/tmp')

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchtext import data
    import random,pickle
    import numpy as np

    from io import BytesIO

    #import spacy,
    import en_core_web_sm
    print('### Using Torch version :',torch.__version__)
except Exception as e:
    print('### Exception occurred while importing modules : {}'.format(str(e)))

# define env variables if there are not existing
S3_BUCKET   = os.environ['MODEL_BUCKET_NAME'] if 'MODEL_BUCKET_NAME' in os.environ else 'eva4p2bucket1'
MODEL_PATH  = os.environ['MODEL_FILE_NAME_KEY'] if 'MODEL_FILE_NAME_KEY' in os.environ else 's9_wordembeddings.pt'
MODEL_obj  = os.environ['MODEL_FILE_OBJ'] if 'MODEL_FILE_OBJ' in os.environ else 's9_wordemb_text_vocab_1.pkl'
print('### S3 Bkt is : {} \nModel path is : {}'.format(S3_BUCKET,MODEL_PATH))


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy', batch_first = True)
LABEL = data.LabelField(dtype = torch.float)

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        #text = [batch size, sent len]
        embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        #cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)
        

# Create client to AWS S3
s3 = boto3.client('s3') 
model_obj   = s3.get_object(Bucket = S3_BUCKET, Key = MODEL_obj)
bytestream  = io.BytesIO(model_obj['Body'].read())
print(f'### Model Text Vocab obj Size: {sys.getsizeof(bytestream) // (1024 * 1024)}')
#with open('/content/text_vocab_1.pkl','rb') as p:
txt_vocab = pickle.load(bytestream)
print('### type of text vocab : {}'.format(type(txt_vocab))) 
INPUT_DIM = len(txt_vocab) #len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3,4,5]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = txt_vocab.stoi[TEXT.pad_token] #TEXT.vocab.stoi[TEXT.pad_token]


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'### Device is : {device}')

def load_model_from_s3bkt():
    try:
        obj         = s3.get_object(Bucket = S3_BUCKET, Key = MODEL_PATH)
        bytestream  = io.BytesIO(obj['Body'].read())
        print('### Loading model...')
        print(f'Model Size: {sys.getsizeof(bytestream) // (1024 * 1024)}')
        model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
        model = model.to(device)
        model.load_state_dict(torch.load(bytestream,map_location=torch.device('cpu')))
        print('### Model in eval mode : {}'.format(model.eval()))
        
        print('### Model is loaded and returning model')
        return model
    except Exception as e:
        print('### Exception in loading a model : {}'.format(str(e)))
        raise(e)

model   = load_model_from_s3bkt()


nlp = spacy.load('en')
def predict_sentiment(model, sentence, min_len = 5):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    #print(tokenized)
    indexed = [txt_vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

def cnn_sentiment_pred(event, context):
    try:
        print('### You are in handler dcGAN_car function')
        print('### event is : {}'.format(event))
        print('### Context is : {}'.format(context))
        
        body = {
            "message": 'Negative' if predict_sentiment(model, "This film is terrible")<0.5 else 'Possitive',
            "input": event
        }

        response = {
            "statusCode": 200,
            "body": json.dumps(body)
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
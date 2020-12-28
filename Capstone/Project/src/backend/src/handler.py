print("Import START...")
try:
    import unzip_requirements
    print("unzip requirements imported")
    from requests_toolbelt.multipart import decoder
    print("decoder imported")
    import json
    import base64
    import os
    import urllib
    import datetime
    import boto3
    print("basic imported")

    import torch
    from torchtext import data
    import random
    import nltk
    nltk.download('stopwords')
    SEED = 1234
    from nltk.corpus import stopwords
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    TEXT = data.Field(tokenize='spacy', stop_words=stopwords.words('english'))
    LABEL = data.LabelField(dtype=torch.float)    
except ImportError:
    print('### Exception occured in import')

print('### Import End....')


BUCKET_NAME = os.environ['BUCKET_NAME'] if 'BUCKET_NAME' in os.environ else 'aiendeavour'
print(BUCKET_NAME)
LOCAL_FILE_PATH = ''

s3 = boto3.resource(u's3')

def predict_sentiment(model, sentence, min_len = 5):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    #indexed = [data.stoi[t] for t in tokenized]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

def upload(event, context):
    try:
        print(json.dumps(event))
        content_type_header = event['headers']['content-type']        
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

def train(event, context):
    try:
        print(LOCAL_FILE_PATH)
        fields = [('text', TEXT), ('label', LABEL)]
        train_data = data.TabularDataset.splits(
                                                path = '',
                                                train = LOCAL_FILE_PATH,
                                                format = 'csv',
                                                fields = fields,
                                                skip_header = True
        )
        train_data = train_data[0]
        print(vars(train_data.examples[0]))
        train_data, test_data = train_data.split(split_ratio=0.7, random_state=random.seed(5))
        train_data, valid_data = train_data.split(split_ratio=0.7,random_state=random.seed(2))
        TEXT.build_vocab(train_data, max_size=10000)
        LABEL.build_vocab(train_data)

        BATCH_SIZE = 64
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=BATCH_SIZE,
            sort_key=lambda x:len(x.text),
            sort_within_batch=False,
            device=device)

        import torch.nn as nn
        class RNN(nn.Module):
            def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
                super().__init__()
                self.embedding = nn.Embedding(input_dim, embedding_dim)
                self.rnn = nn.RNN(embedding_dim, hidden_dim)
                self.fc = nn.Linear(hidden_dim, output_dim)
            def forward(self, text):
                #text = [sent len, batch size]
                embedded = self.embedding(text)
                output, hidden = self.rnn(embedded)
                #output = [sent len, batch size, hid dim]
                #hidden = [1, batch size, hid dim]
                assert torch.equal(output[-1,:,:], hidden.squeeze(0))
                return self.fc(hidden.squeeze(0))
        INPUT_DIM = len(TEXT.vocab)
        EMBEDDING_DIM = 100
        HIDDEN_DIM = 512
        OUTPUT_DIM = 1
        model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

        import torch.optim as optim
        optimizer = optim.SGD(model.parameters(), lr=1e-3)

        criterion = nn.BCEWithLogitsLoss()
        model = model.to(device)
        criterion = criterion.to(device)
        def binary_accuracy(preds, y):
            rounded_preds = torch.round(torch.sigmoid(preds))
            correct = (rounded_preds == y).float() #convert into float for division
            acc = correct.sum()/len(correct)
            return acc

        def train(model, iterator, optimizer, criterion):
    
            epoch_loss = 0
            epoch_acc = 0
            
            model.train()
            
            for batch in iterator:
                
                optimizer.zero_grad()
                        
                predictions = model(batch.text).squeeze(1)
                
                loss = criterion(predictions, batch.label)
                
                acc = binary_accuracy(predictions, batch.label)
                
                loss.backward()
                
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                
            return epoch_loss / len(iterator), epoch_acc / len(iterator)    
        def evaluate(model, iterator, criterion):
            
            epoch_loss = 0
            epoch_acc = 0
            
            model.eval()
            
            with torch.no_grad():
            
                for batch in iterator:

                    predictions = model(batch.text).squeeze(1)
                    
                    loss = criterion(predictions, batch.label)
                    
                    acc = binary_accuracy(predictions, batch.label)

                    epoch_loss += loss.item()
                    epoch_acc += acc.item()

            return epoch_loss / len(iterator), epoch_acc / len(iterator)

        import time

        def epoch_time(start_time, end_time):
            elapsed_time = end_time - start_time
            elapsed_mins = int(elapsed_time / 60)
            elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
            return elapsed_mins, elapsed_secs
                

        N_EPOCHS = 5

        best_valid_loss = float('inf')

        for epoch in range(N_EPOCHS):

            start_time = time.time()
            
            train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
            valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
            
            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'tut1-model.pt')
            
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

            model.load_state_dict(torch.load('tut1-model.pt'))

            test_loss, test_acc = evaluate(model, test_iterator, criterion)

            print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
            
        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({"Test accuracy": test_acc})
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

def predict(event, context):
    try:
        print(json.dumps(event))
        requestBody = base64.b64decode(event["body"])
        print(requestBody)
        body = json.loads(requestBody)
        print(json.dumps(body))
        input = body["inputtext"]
        print(input)        

        sentiment = "positive"
        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({"input": input , "predictionValue":0.01, "sentiment": sentiment })
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

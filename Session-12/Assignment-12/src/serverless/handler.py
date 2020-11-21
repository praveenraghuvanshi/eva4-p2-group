try:
    print('### Import started')
    import unzip_requirements
    import json,boto3,os,tarfile,io,base64,json,pickle
    from requests_toolbelt.multipart import decoder

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


    import spacy
    import de_core_news_sm

    import dill
    import random
    import numpy as np
    
    
    #from modelsUtility import *
    print('### Import End....')
except ImportError:
    print('### Exception occurred in import')


device = 'cpu'
DEVICE = 'cpu'
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"    
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
LOWER = True
num_words = 11

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)


##############
# Define classes and methods:
##############
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator
        
    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)
    
    def encode(self, src, src_mask, src_lengths):
        return self.encoder(self.src_embed(src), src_mask, src_lengths)
    
    def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
               decoder_hidden=None):
        return self.decoder(self.trg_embed(trg), encoder_hidden, encoder_final,
                            src_mask, trg_mask, hidden=decoder_hidden)
        

class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout)
        
    def forward(self, x, mask, lengths):
        """
        Applies a bidirectional GRU to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        output, final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # we need to manually concatenate the final states for both directions
        fwd_final = final[0:final.size(0):2]
        bwd_final = final[1:final.size(0):2]
        final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2*dim]

        return output, final


class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""
    
    def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5,
                 bridge=True):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
                 
        self.rnn = nn.GRU(emb_size + 2*hidden_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
                 
        # to initialize from the final encoder state
        self.bridge = nn.Linear(2*hidden_size, hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2*hidden_size + emb_size,
                                          hidden_size, bias=False)
        
    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        context, attn_probs = self.attention(
            query=query, proj_key=proj_key,
            value=encoder_hidden, mask=src_mask)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        
        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output
    
    def forward(self, trg_embed, encoder_hidden, encoder_final, 
                src_mask, trg_mask, hidden=None, max_len=None):
        """Unroll the decoder one step at a time."""
                                         
        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)
        
        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)
        
        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []
        
        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
              prev_embed, encoder_hidden, src_mask, proj_key, hidden)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        return torch.tanh(self.bridge(encoder_final))            


class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""
    
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()
        
        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        
        # to store attention scores
        self.alphas = None
        
    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)
        
        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)
        
        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float('inf'))
        
        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas        
        
        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)
        
        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas

def greedy_decode(model, src, src_mask, src_lengths, max_len=100, sos_index=1, eos_index=None):
    """Greedily decode a sentence."""
    print(f'### In greedy_decode method, type of model is : {type(model)}')
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

def make_model(src_vocab, tgt_vocab, emb_size=256, hidden_size=512, num_layers=1, dropout=0.1):
    "Helper: Construct a model from hyperparameters."

    attention = BahdanauAttention(hidden_size)

    model = EncoderDecoder(
        Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout),
        nn.Embedding(src_vocab, emb_size),
        nn.Embedding(tgt_vocab, emb_size),
        Generator(hidden_size, tgt_vocab))

    return model.cuda() if torch.cuda.is_available() else model

# Make Prediction
def translate_german2english(model, sentence,spacy_model,src_fields,DEVICE='cpu'):
    print('### In translate_german2english method')
    model.eval()
    tokenized = [tok.text for tok in spacy_model.tokenizer(sentence)]
    print(f'### tokenized : {tokenized}')
    indexed = [src_fields.vocab.stoi[t] for t in tokenized]
    print(f'### indexed : {indexed}')
    tensor = torch.LongTensor(indexed).to(DEVICE)
    tensor = tensor.unsqueeze(0)
    print(f'### tensor : {tensor}')
    print(f'### Tensor size is : {tensor.size()}')
    pad_index = 0
    src_mask = (tensor != pad_index).unsqueeze(0).to(DEVICE)
    print(f'### src_mask : {src_mask} and src_mask size is : {src_mask.size()}')
    src_lengths = torch.LongTensor([tensor.size()[1]]).to(DEVICE)
    print(f'### src_lengths : {src_lengths} and size is : {src_lengths.size()}')
    result, _ = greedy_decode(
          model, tensor, src_mask, src_lengths,
          max_len=25, sos_index=TRG.vocab.stoi[SOS_TOKEN], eos_index=TRG.vocab.stoi[EOS_TOKEN])
    
    output = lookup_words(result,vocab= TRG.vocab)
    outText = ' '.join(output)
        
    return outText

'''
S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'suman-p2-bucket'
#MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 's11-german2EnglishText-encoder-decoder-full-cpu-stdict.pt'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 's11-german2EnglishText-encoder-decoder-full-cpu.pt'
SRC_TEXT_FIELD= os.environ['SRC_TEXT_FIELD'] if 'SRC_TEXT_FIELD' in os.environ else 'SRC_fields.pkl'
TRG_TEXT_FIELD= os.environ['TRG_TEXT_FIELD'] if 'TRG_TEXT_FIELD' in os.environ else 'TRG_fields.pkl'
'''

S3_BUCKET  = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'suman-p2-bucket'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 's11-german2EnglishText-encoder-decoder-full-cpu-stdict.pt'
SRC_FIELD  = os.environ['SRC_FIELD'] if 'SRC_FIELD' in os.environ else 'SRC_fields.pkl'
TRG_FIELD  = os.environ['TRG_FIELD'] if 'TRG_FIELD' in os.environ else 'TRG_fields.pkl'
print(f'### model path : {MODEL_PATH} SRC : {SRC_FIELD} TRG : {TRG_FIELD}')

s3 = boto3.client('s3')

##############
#Load SRC and TRG pkl files:
##############
if os.path.isfile(SRC_FIELD) != True:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=SRC_FIELD)
    print("### Main: Loading SRC pkl file in Bytestream...")
    bytesstream_SRC = io.BytesIO(obj['Body'].read())
    SRC = torch.load(bytesstream_SRC, pickle_module=dill)


if os.path.isfile(TRG_FIELD) != True:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=TRG_FIELD)
    print("### Main: Loading TRG pkl file in Bytestream...")
    bytesstream_TRG = io.BytesIO(obj['Body'].read())
    TRG = torch.load(bytesstream_TRG, pickle_module=dill)

    
################
criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
model = make_model(len(SRC.vocab), len(TRG.vocab),emb_size=256, hidden_size=256,num_layers=1, dropout=0.2)
##############
#Load Model file:
##############
print(f'### Check dirs before we do import model : {os.listdir()}')
if os.path.isfile(MODEL_PATH) != True:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
    bytesstream_model = io.BytesIO(obj['Body'].read())
    chkpnt = torch.load(bytesstream_model,map_location=torch.device('cpu'))
    print("### Main: Model Loaded...")
    
model.load_state_dict(chkpnt)

print(f'### Device is : {device}')
model = model.to(device)
model.eval()

try:
    print('### loading spacy')
    os.system('cp -r de_core_news_sm* /tmp/pkgs-from-layer/')
    spacy_de = spacy.load('/tmp/pkgs-from-layer/de_core_news_sm/de_core_news_sm-2.2.5')
    print('### spacy de model is loaded')
except Exception as e:
    print(f'### Exception occured in loading spacy model :{str(e)}')


def translateg2e(event, context):
    try:
        print('### You are in translateg2e method')
        print(f'### Event is : {event}')
        bodyTemp = event["body"]
        body = json.loads(bodyTemp)
        print(body,type(body))
        germanText = body["germanText"]
        print(f'### germanText : {germanText} and its type is : {type(germanText)}')

        predValue = translate_german2english(model,germanText,spacy_de,SRC)
        print('### Predicted value',predValue)
        response = {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({"input": germanText , "output":predValue})
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
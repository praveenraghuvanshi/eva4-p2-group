try:
    print('### Import started')
    import os
    import io
    import base64
    import json
    import sys
    import unzip_requirements
    import requests_toolbelt
    # from requests_toolbelt.multipart import decoder
    print('### Imported unzip_requirements')

    import torch
    import torch.nn.functional as F
    print('### Import torch completed')

    import torchvision.transforms as transforms    
    print('### Imported torchvision completed')

    import boto3
    print('### Imported boto3')

    import numpy as np
    from PIL import Image
    print('### Imported numpy and PIL')

    print('### Import End....')
except ImportError:
    print('### Exception occurred in import')

S3_BUCKET  = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'eva4p2bucket1'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'model-caption.pth.tar'
print(f'### model path : {MODEL_PATH}')

##############
# Define classes and methods:
##############
def caption_image_beam_search(encoder, decoder, image, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """
    print('Inside caption_image_beam_search')

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = transform(image)  # (3, 256, 256)
    print(image.shape)

    # Encode
    image = image.unsqueeze(0).to(device)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas

# Generate Caption
def generateCaptionInternal(image):

    print('Inside generateCaptionInternal')
    # Load word map (word2ix)
    with open('WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json', 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, image, word_map, 5)
    alphas = torch.FloatTensor(alphas)

    # create sentence
    words = [rev_word_map[ind] for ind in seq]
    print(words)

    # using list comprehension 
    caption = ' '.join([str(elem) for elem in words])
    print(caption)
    return caption

##############
#Load Model file:
##############
def load_model_from_s3():
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print('Creating Bytestream')
        bytestream = io.BytesIO(obj['Body'].read())
        print('Loading model...')
        print(sys.getsizeof(bytestream) // (1024 * 1024))
        model = torch.load(bytestream, map_location=torch.device('cpu'))
        print(model is None)
        print("#### Model Loaded")
        return model
    except Exception as e:
        print('Exception in loading a model')
        print(repr(e))
        raise(e)

#######################
# Execute Initial code:
#######################
s3 = boto3.client('s3')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

model = load_model_from_s3()
decoder = model['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = model['encoder']
encoder = encoder.to(device)
encoder.eval()
print("#### Model Loaded with encoder and decoders")


def generatecaption(event, context):
    try:
        print('### You are in generatecaption method')
        print(f'### Event is : {event}')
        content_type_header = event['headers']['content-type']
        body = base64.b64decode(event["body"])
        print('BODY Loaded')

        picture = requests_toolbelt.multipart.decoder.MultipartDecoder(body, content_type_header).parts[0]
        input_image = Image.open(io.BytesIO(picture.content))
        print('Image loaded')
        print(type(input_image))
        caption = generateCaptionInternal(input_image)
        
        start_tag = '<start>'
        end_tag = '<end>'

        if caption.startswith(start_tag):
            caption = caption[len(start_tag):]

        if caption.endswith(end_tag): 
            caption = caption[:-(len(end_tag))] 

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

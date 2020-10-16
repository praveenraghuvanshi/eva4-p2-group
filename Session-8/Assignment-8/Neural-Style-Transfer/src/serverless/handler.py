try:
    print("### Import START...")
    import unzip_requirements
    from requests_toolbelt.multipart import decoder
    
    import base64, os, io, json, copy, sys
    import boto3

    from PIL import Image
    from io import BytesIO
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import torchvision.models as models
    import torchvision.transforms as transforms
    import numpy as np

    print("Import END...")
except Exception as e:
    print('### Exception occured while importing modules : {}'.format(str(e)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
print(imsize)

# define env variables if there are not existing
S3_BUCKET   = os.environ['MODEL_BUCKET_NAME'] if 'MODEL_BUCKET_NAME' in os.environ else 'eva4p2bucket1'
MODEL_PATH = os.environ['MODEL_FILE_NAME_KEY'] if 'MODEL_FILE_NAME_KEY' in os.environ else 'model-nst.pt'

# load the S3 client when lambda execution context is created
s3 = boto3.client('s3')

def load_model_from_s3():
    try:
        loaded_model = None
        cuda = torch.cuda.is_available()
        print("CUDA Available?", cuda)

        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print('Creating Bytestream')
        bytestream = io.BytesIO(obj['Body'].read())
        print('Loading model...')
        print(f'Model Size: {sys.getsizeof(bytestream) // (1024 * 1024)}')
        # model = torch.jit.load(bytestream, map_location=torch.device('cpu'))
        loaded_model = models.vgg16(pretrained=False).features.to('cpu').eval()
        model_test = torch.load(MODEL_PATH)
        loaded_model.load_state_dict(model_test)
        print(loaded_model is None)
        print('Model Loaded')
        return loaded_model.eval()
    except Exception as e:
        print('Exception in loading a model')
        print(repr(e))
        raise(e)

model = load_model_from_s3()
print(model is None)

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def image_loader(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1) # torch.tensor(mean).view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1) # torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = load_model_from_s3(); # copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d): # if layer.original_name.startswith('Conv2d'):
            i += 1
            name = 'conv_{}'.format(i)
        elif  isinstance(layer, nn.ReLU): # layer.original_name.startswith('ReLU'):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif  isinstance(layer, nn.MaxPool2d): # layer.original_name.startswith('MaxPool2d'):
            name = 'pool_{}'.format(i)
        elif  isinstance(layer, nn.BatchNorm2d): # layer.original_name.startswith('BatchNorm2d'):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        print(f'Inside while loop : {run}')
        def closure():
            print(f'Inside Closure : {run}')
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            print(f'Running model : {run}')
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            print(f'Backpropagation : {loss}')
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        print(f'Before Optimizer : {run}')
        optimizer.step(closure)
        print(f'After Optimizer : {run}')
        print(f'run - {run[0]}, steps - {num_steps}')

    # a last correction...
    input_img.data.clamp_(0, 1)
    print(input_img.shape)

    return input_img

def getneuralstyletransfer(event, context):
    try:
        print('### You are in handler getneuralstyletransfer function')
        print('### event is   : {}'.format(event))
        print('### Context is : {}'.format(context))
        
        content_type_header = event['headers']['content-type']
        body    = base64.b64decode(event["body"])
        print(f'### Body is : {body}')
        picture1 = decoder.MultipartDecoder(body, content_type_header).parts[0]
        picture2 = decoder.MultipartDecoder(body, content_type_header).parts[1]
        
        image_data1 = picture1.content
        image_data2 = picture2.content

        print(f'### Image - 1 : {len(image_data1)}')
        print(f'### Image - 2 : {len(image_data2)}')
        
        img1 = Image.open(io.BytesIO(image_data1))
        img2 = Image.open(io.BytesIO(image_data2))        
        
        img_t1 = transforms.ToTensor()(img1)
        img_t2 = transforms.ToTensor()(img2)

        print(f'### img-1 tensor : {img_t1.shape}')
        print(f'### img-2 tensor : {img_t2.shape}')

        # ******
        content_img = image_loader(image_data1)
        style_img = image_loader(image_data2)
        input_img = content_img.clone()

        assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

        print(style_img.shape)

        output = run_style_transfer(model, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img, num_steps=30)
        print(output.shape)
        # ******

        temp_img = output[0].detach()
        npImg = temp_img.permute(1, 2, 0).numpy()
            
        pil_img = Image.fromarray((npImg * 255).astype(np.uint8))
        buff = BytesIO()

        pil_img.save(buff, format="JPEG")
        new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
        img_str = f"data:image/jpeg;base64,{new_image_string}"
        print(f'### Final Image String is : {img_str}')

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({'imagebytes': img_str})
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
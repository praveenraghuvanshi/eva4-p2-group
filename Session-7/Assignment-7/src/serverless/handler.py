print("Import START...")
try:
    import unzip_requirements
except ImportError:
    pass
    
from requests_toolbelt.multipart import decoder

import torch
import torch.nn as nn

import os
import io
import base64
import json
import sys
print("Import END...")


# define env variables if there are not existing
MODEL_PATH = 'netG_chkpt_1420.pth'

# Define generator
class Generator(nn.Module):
    """
    Creates the Generator

    nz (int): size of the latent z vector
    ngf (int): number of feature maps for the generator
    """
    def __init__(self, nz: int = 64, ngf: int = 64):
        super(Generator, self).__init__()
        #self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
        
nz = 64;ngf = 64;nc = 3
netG = Generator(nz, ngf).to(device)        

def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min + 1e-5)

def norm_tensor(t):
    norm_ip(t, float(t.min()), float(t.max()))
    

def dcGAN_car(event, context):
    try:
        content_type_header = event['headers']['content-type']
        print('### Load Model : {}'.format(netG.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))))

        with torch.no_grad():
            fake = netG(torch.randn(1, 64, 1, 1).to(device)).detach().cpu()
            
        fakei = fake[0]
        norm_tensor(fakei)
        print('### fakei is :',fakei)
        
        s6Img = fakei.numpy()
        print('s6Img shape ',s6Img.shape)
        img_chw = s6Img.reshape(64,64,3)
        img_chw = np.float32(img_chw)/255.0
        print('img_chw shape',img_chw.shape)
        type(img_chw)
        img_str = f"data:image/jpeg;base64,{base64.b64encode(img_chw.tobytes())}"

        print(img_str)
        
        print('BODY Loaded')

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
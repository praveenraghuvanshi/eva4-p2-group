import io
import random, decimal
import torch
import torchvision.transforms as transforms
from PIL import Image

def transform_image(imagePath):
    print('Transformation Start...')
    try:
        transformations = transforms.Compose([
            transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = Image.open(imagePath)
        print('Transformation finished...')
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print('Exception in transforming an image')
        print(repr(e))
        raise(e)

def train_model():
    print('Training Image classification')
    randomAccuracy = float(decimal.Decimal(random.randrange(50, 80))/100)
    print(randomAccuracy)
    return {
        "test_loss" : 0.15,
        "test_acc" : randomAccuracy,
        "model" : 'model.pt',
        "model_url" : 'http://0.0.0.0:80'
    }

def classify_image(imagePath, modelPath):
    model = torch.jit.load(modelPath, map_location=torch.device('cpu'))
    print(model is None)

    tensor = transform_image(imagePath)
    result = model(tensor).argmax().item()

    print(f'Predicted value: {result}')
    return result

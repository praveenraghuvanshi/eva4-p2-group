import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.transforms import Resize


MODEL_FILE = "mobilenet_v2.pt"
BATCH_SIZE = 4
DATASET_SPLIT = 0.7
LEARNING_RATE = 0.001
EPOCHS = 10

class_to_id: any

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

# Calculate Mean and Std
def calcMeanStd(dl):
    mean = 0.
    std = 0.
    nb_samples = 0.
    print("Calulating MEAN and STD")
    for data,lbl in dl:
        print(data.shape())
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    #print('Mean is \t: {}\nstd is \t: {}\nnb_samples \t:{}'.format(mean,std,nb_samples))
    return {'mean': mean, 'std' : std, 'noOfImgs' : nb_samples}

def train_model(dspth):
    print(dspth)
    import torch
    # model = torch.hub.load('pytorch/vision', MODEL_FILE, pretrained=False)
    model = models.mobilenet_v2(pretrained=False)
    print("Model Loaded")
    import torchvision,PIL
    import numpy as np
    
    SEED = 1234
    dspth = dspth

    # 1. Load Dataset
    transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(size=(128, 128)),
                    torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
                    torchvision.transforms.ToTensor()
                ])
    ds = torchvision.datasets.ImageFolder(dspth, transform=transform)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE,shuffle=True)
    print(f'### Dataset classes are : {ds.classes}')
    print(f'### Dataloader shape : {dl}')
    
    cuda = torch.cuda.is_available()
    print('### cuda Available or not ? : {}'.format(cuda))
    dataloader_args = dict(shuffle=True, batch_size=BATCH_SIZE, num_workers=1, pin_memory=True) if cuda else dict(shuffle=True, batch_size = BATCH_SIZE)

    trainds, testds = torch.utils.data.random_split(dataset=ds, lengths=[len(ds)-int(DATASET_SPLIT*len(ds)), int(DATASET_SPLIT*len(ds)) ])
    print('### Length of Train DS is : {} and Length of Test DS is  : {}'.format(len(trainds), len(testds)))
    
    traindl = torch.utils.data.DataLoader(trainds,  **dataloader_args)
    testdl  = torch.utils.data.DataLoader(testds, **dataloader_args)

    print('### Dataset contains {} - classes : {}'.format(len(ds.classes),dl.dataset.class_to_idx))
    print(type(dl.dataset.class_to_idx))
    # print(f'### Train DL mean and std : {calcMeanStd(traindl)}')
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.classifier[1]

    for param in model.parameters():
        param.requires_grad = False    

    n_inputs = model.classifier[1].in_features
    print(n_inputs)

    model.classifier = torch.nn.Sequential(
                            torch.nn.Linear(n_inputs, 512),
                            torch.nn.Linear(512, len(ds.classes)),
                            torch.nn.LogSoftmax(dim=1)
                        )
    model.to(device)
    # Loss
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr = LEARNING_RATE)
    print(model)
    
    def train_model(model, batch_size, n_epochs):
        train_losses,test_losses  = [],[]
        avg_train_losses, avg_test_losses = [],[]
        total, correct, wrong, acc = 0,0,0,0
        misclass_imgList = []
        misclass_imgList = np.array(misclass_imgList)
        for epoch in range(1,n_epochs+1):
            print(f'### Trainig is in progress for epoch : {epoch}')
            ##### Trainnig:
            model.train()
            for inputs, target in traindl:
                inputs, target = inputs.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(inputs)
                loss   = criterion(output,target)
                optimizer.step()
                train_losses.append(loss.item())
            
            ##### Test model:
            model.eval()
            with torch.no_grad():
                for inputs,target in testdl:
                    inputs, target = inputs.to(device), target.to(device)
                    output = model(inputs)
                    _, predicted = torch.max(output.data, 1)
                    loss = criterion(output,target)
                    test_losses.append(loss.item())
                    
                    total   += target.size(0)
                    correct += (predicted == target).sum().item()
                    wrong   += (predicted != target).sum().item()
                    print('### Total : {}, corrct : {} and wrong : {}'.format(total,correct,wrong))
                    misclass_imgList = np.append(misclass_imgList, ((predicted==target)==False).nonzero().view(-1).numpy())
                    #import pdb;pdb.set_trace()
        
                
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            test_loss  = np.average(test_losses)
            avg_train_losses.append(train_loss)
            avg_test_losses.append(test_loss)
            test_acc =  (correct / total) * 100.0
            
            epoch_len = len(str(n_epochs))
            print_msg = (f'\n[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.4f} ' +
                         f'\ttest_loss: {test_loss:.4f}' +
                         f'\tTest_accuracy: {test_acc:.2f}')
            print(print_msg)
            
        return  model, avg_train_losses, avg_test_losses,misclass_imgList,total,correct,wrong, test_acc
                
    model, train_loss, test_loss,misclass_imgList,total,correct,wrong,test_acc = train_model(model, batch_size=BATCH_SIZE, n_epochs=EPOCHS)
    print(f"Accuracy: {test_acc}")
    traced_model = torch.jit.trace(model, torch.rand(BATCH_SIZE, 3,128,128))
    torch.jit.save(traced_model, MODEL_FILE)
    return (MODEL_FILE, test_loss, test_acc, dl.dataset.class_to_idx)


def classify_image(imagePath, modelPath):
    model = torch.jit.load(modelPath, map_location=torch.device('cpu'))
    print(model is None)

    tensor = transform_image(imagePath)
    result = model(tensor).argmax().item()

    print(f'Predicted value: {result}')
    return result

# imports here
import argparse

import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from collections import OrderedDict
# torchvision.datasets import ImageFolder
from torch.autograd import Variable
import numpy as np
from PIL import Image
print("Stop 1 - after imports")

def parse_args():
    print("Parsing..")
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir', metavar='data_dir', type=str)
    parser.add_argument('--arch', dest='arch', default='densenet121', choices=['densenet121', 'vgg16'])
    parser.add_argument('--save_dir', dest='save_dir', action='store', default="checkpoint.pth")
    parser.add_argument('--hidden_units', action='store', dest='hidden_units', default='512')
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', default='0.001')
    parser.add_argument('--epochs', dest='epochs', default='3')
    #if it fails maybe change the name for arch based on checkpoint save or add comment back
    
    return parser.parse_args()

def train_mode(model, criterion, optimizer, trainloader, epochs, validloader):
    print("About to train")
    print_every = 10
    steps = 0
    #set the device
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    start=time.time()
    print("Start training: ")
    
    for e in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1 
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                validation_loss = 0
                accuracy=0
                
                for inputs2, labels2 in validloader:
                    optimizer.zero_grad()
                    inputs2, labels2 = inputs2.to(device), labels2.to(device)
                    
                    with torch.no_grad():
                        
                        outputs = model.forward(inputs2)
                        validation_loss = criterion(outputs,labels2)
                        
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        
                        equals = top_class == labels2.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            

                validation_loss = validation_loss / len(validloader)
                accuracy = accuracy /len(validloader)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Training Loss: {:.4f}".format(running_loss/print_every),
                  "Validation Loss {:.4f}".format(validation_loss),
                  "Accuracy: {:.4f}".format(accuracy),
                 )

                running_loss = 0
                model.train()
                
    time_elapsed=time.time() - start
    print("\nTime spent training: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))
    return model
                    
def save_checkpoint(model, optimizer, path, output_size, train_datasets, classifier, args):
    print("Saving model...")
    model.class_to_idx = train_datasets.class_to_idx
    
    #hidden units and input size pot issues - added args
    checkpoint={'pretrained_model': args.arch,
           # 'input_size': input_size,
            'output_size': output_size,
            'state_dict': model.state_dict(),
            'classifier': model.classifier,
            'class_to_idx': model.class_to_idx,
            'optimizer': optimizer.state_dict(),
            'epochs': args.epochs,
            'hidden_units': args.hidden_units,
            'learning_rate': args.learning_rate
           }
    torch.save(checkpoint, 'checkpoint.pth')
    print("Model Saved.")

def main():
    
    print("in Main Module...")
    args = parse_args()
    
    data_dir='flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    test_valid_transforms = transforms.Compose([transforms.Resize(224),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_transforms=transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                                                             
    # Load the datasets with ImageFolder
    train_datasets= datasets.ImageFolder(train_dir, transform=train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_valid_transforms)
    valid_datasets =  datasets.ImageFolder(valid_dir, transform=test_valid_transforms)
        
    # Using the image datasets and the trainforms, define the dataloaders
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    testloaders= torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle= True)
    validloaders=torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)
    
    #load the model
    model=getattr(models, args.arch)(pretrained=True)
    
    if args.arch == 'vgg16':
        model=models.vgg16(pretrained=True)
        classifier=nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 1024)),
            ('dropout', nn.Dropout(p=0.5)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(1024, 102)),
            ('output', nn.LogSoftmax(dim=1))]))
    elif args.arch == 'densenet121':
        model=models.densenet121(pretrained=True)
        #this portion distinct from statement blocks
        classifier = nn.Sequential(OrderedDict([ 
                          ('fc1', nn.Linear(1024, 500)),
                          ('dropout', nn.Dropout(p=0.6)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(500, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))                         
        
    #turn off gradient
    for param in model.parameters():
        param.requires_grad=False
        
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    epochs=int(args.epochs)
    class_idx=train_datasets.class_to_idx
    output_size=102
    train_mode(model, criterion, optimizer, trainloaders, epochs, validloaders)
    model.class_to_idx=class_idx
    path=args.save_dir
    save_checkpoint(model, optimizer, path, output_size, train_datasets, classifier, args)
    
    #prev from savecheck function(model, epochs, learning_rate, optimizer, input_size, file_path, output_size, train_datasets, hidden_units, arch):
   
    
if __name__ == "__main__":
    main()
    
                    
                    
                    
        

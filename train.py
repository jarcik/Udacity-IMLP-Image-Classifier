import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
from PIL import Image
import numpy as np
import seaborn as sns
import argparse
from workspace_utils import active_session

#arguments from console
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory", type=str)
    parser.add_argument("--save_dir", type = str, default = "./")
    parser.add_argument("--arch", type = str, default = "vgg16")
    parser.add_argument("--learning_rate", type = float, default = 0.001)
    parser.add_argument("--hidden_units", type = int, default = 1024)
    parser.add_argument("--epochs", type = int, default = 5)
    parser.add_argument("--gpu", type = bool, default = False)
    return parser.parse_args()

#transforms of image
def get_transforms(train = False):
    transform = None
    crop = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if train:
        transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(crop),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)])
    else:
        transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(crop),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean, std)])
    return transform

#loaders for data set
def create_loaders(data_path):
    if not data_path:
        print("Not a valid path for data")
        
    #images
    train_data = datasets.ImageFolder(data_path + '/train', transform = get_transforms(True))
    valid_data = datasets.ImageFolder(data_path + '/valid', transform = get_transforms())
    test_data = datasets.ImageFolder(data_path + '/test', transform = get_transforms())

    #data loaders
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size)
    
    return train_loader, valid_loader, test_loader, train_data.class_to_idx

#pretrained model
def get_model(model_name, hidden_units):
    model_dic = {
        "vgg13" : models.vgg13,
        "vgg16" : models.vgg16,
        "densenet121" : models.densenet121
    }
    
    if not model_name:
        print("Enter valid model_name: {}".format(', '.join(model_dic.keys())))
              
    model_fce = model_dic.get(model_name, None)    
    if not model_fce:
        print("Please, try another model from the menu: {}".format(', '.join(model_dic.keys())))
    
    model = model_fce(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))    
    model.classifier = classifier
    
    return model

#validation in training process
def validation(model, valid_loader, criterion, device):
    loss = 0
    accuracy = 0
    
    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        loss += criterion(output, labels).item()

        ps = torch.exp(output)
        
        top_prob, top_label = ps.topk(1, dim = 1)                
        equals = top_label == labels.view(*top_label.shape)
        accuracy = accuracy + torch.mean(equals.type(torch.FloatTensor)).item()
    
    return loss, accuracy

#train model
def train(epochs, train_loader, device, optimizer, model, criterion, valid_loader):
    steps = 0
    running_loss = 0
    print_every = 5
    
    print("Started training \n")
              
    with active_session():
        for e in range(epochs):
            for inputs, labels in train_loader:

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)

                loss = criterion(logps, labels)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()
                steps += 1

                if steps % print_every == 0:
                    model.eval()

                    with torch.no_grad():
                        valid_loss, accuracy = validation(model, valid_loader, criterion, device)

                    print(f"Epoch {e+1}/{epochs}: "
                          f"Training loss: {running_loss/print_every:.3f}; "
                          f"Valdation loss: {valid_loss/len(valid_loader):.3f}; "
                          f"Validation accuracy: {accuracy/len(valid_loader):.3f}")

                    running_loss = 0
                    model.train()
            
    print("\n Done traning!")    
    return model

#test model
def test(test_loader, device, model):
    correct, total = 0, 0
    with torch.no_grad ():
        model.eval()
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    print('Accuracy for test data is %d%%'% (100 * correct / total))

#save model
def save_checkpoint(model, class_to_idx, epochs, optimizer, save_dir, name, hidden_units):
    model.class_to_idx = class_to_idx

    checkpoint = {'classifier' : model.classifier,
                  'state_dict' : model.state_dict(),
                  'class_to_idx' : model.class_to_idx,
                  'epochs' : epochs,
                  'optimizer_state' : optimizer.state_dict(),
                  'name' : name,
                  'hidden_units' : hidden_units
                 }        

    torch.save(checkpoint, save_dir + "/checkpoint.pth")
    print("Model saved!")
    
#main
def main():
    args = arg_parse()
    
    train_loader, valid_loader, test_loader, class_to_idx = create_loaders(args.data_directory)    
    model = get_model(args.arch, args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)    
    device = torch.device("cuda:0" if args.gpu else "cpu")
    model.to(device)
    
    trained_model = train(args.epochs, train_loader, device, optimizer, model, criterion, valid_loader)
    test(test_loader, device, trained_model)
    save_checkpoint(trained_model, class_to_idx, args.epochs, optimizer, args.save_dir, args.arch, args.hidden_units)

#run it
if __name__ == '__main__': main()

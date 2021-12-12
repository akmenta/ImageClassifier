'''
Training program for the ImageClassifier model
This code will be built off of the code we built in the Jupyter notebook
'''

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import sys
import argparse
from PIL import Image
import json

'''
Input args between predict and train
--save_dir
--arch
--learning_rate
--hidden_units
--epochs
--gpu
--category_names
'''

debugFlag = True

def get_input_args():
    
    try:
        
        parser = argparse.ArgumentParser(description='Input args')
        
        parser.add_argument('--save_dir', type=str, action='store', default='.',
                    help='Input Model save dir, default is current dir')
    
        parser.add_argument('--arch', type=str, default="vgg19",
                    help="Input model arch: vgg19 ...etc, default vgg19")
        
        parser.add_argument('--learning_rate', default=0.003,
                    help='Input learning rate, default is 0.003')
                            
        parser.add_argument('--hidden_units', default=4096,
                    help='Input hidden units, default is 4096')
                            
        parser.add_argument('--epochs', default=6,
                    help='Input epochs, default is 6')
    
        parser.add_argument('--gpu', default=True,
                    help='Use GPU or CPU to train model: True for use GPU, default is True')            
   
        parser.add_argument('--cat_to_name', action='store', default = 'cat_to_name.json',
                    help='Enter mapping of categories to real names. Default is cat_to_name.json')

        args = parser.parse_args() 
       
        if(debugFlag):
            print(f'get_input_args: args: {args}')
        
    except BaseException as exception:
        print("get_input_args !", repr(exception));
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_type, exc_tb.tb_lineno)
        sys.exit()
                            
    return args;


#
# Load the three datasets
#

def load_the_data():
    
    try:
        
        data_dir = 'flowers'
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
        
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

        valid_test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

        # Load the datasets with ImageFolder
        train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
        valid_data = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
        test_data = datasets.ImageFolder(test_dir, transform=valid_test_transforms)

        # Using the image datasets and the trainforms, define the dataloaders
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
        validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
        
        return train_data, valid_data, test_data, trainloader, validloader, testloader

    except BaseException as exception:
        print("load_the_data !", repr(exception));
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_type, exc_tb.tb_lineno)
        sys.exit()
        
#
# Map the json file
#
        
def label_mapping(icat_to_name):
      
    try:           
        with open(icat_to_name, 'r') as f:
            cat_to_name = json.load(f)
    
        if(debugFlag):
            print('Cat count: {}'.format(len(cat_to_name)))
            for x in list(cat_to_name)[0:3]:
                print ("key {}, value {} ".format(x,  cat_to_name[x]))
                
        return cat_to_name      

    except BaseException as exception:
        print("label_mapping !", repr(exception));
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_type, exc_tb.tb_lineno)
        sys.exit()

#
# Building  the classifier
#

def build_classifier(iarch, idevice, ilr, ihidden_units):
      
    try:
        
        model =  getattr(models,iarch)(pretrained=True)
        
        if(debugFlag):
            print(f'build_and_train_classifier: model: {model}')
            
        inp_features = model.classifier[0].in_features
        
        if(debugFlag):
            print(f'build_and_train_classifier: inp_features: {inp_features}')
        
        for param in model.parameters():
            param.requires_grad = False
            
        model.classifier = nn.Sequential(nn.Linear(inp_features, ihidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(ihidden_units, 102),
                                 nn.LogSoftmax(dim=1))
                                 
        criterion = nn.NLLLoss()

        optimizer = optim.Adam(model.classifier.parameters(), lr=ilr)

        model.to(idevice);
        
        if(debugFlag):
            print(f'build_and_train_classifier: model: {model}')
        
        return model, optimizer, criterion, inp_features
        
    except BaseException as exception:
        print("build_and_train_classifier:", repr(exception));
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_type, exc_tb.tb_lineno)
        sys.exit()

#
# Training the classifier
#
def train_classifier(imodel, itrain_loader, ivalid_loader, \
                     itest_loader, ioptimizer, icriterion, iepochs, idevice):
    
    try:
        
        epochs = iepochs
        steps = 0
        running_loss = 0
        print_every = 20
        
        for epoch in range(epochs):
            for inputs, labels in itrain_loader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(idevice), labels.to(idevice)
                # Forward pass, then backward pass, then update weights
 
                
                #logps = imodel.forward(inputs)
                logps = imodel(inputs)
                loss = icriterion(logps, labels)
                ioptimizer.zero_grad()
                loss.backward()
                ioptimizer.step()
        
                running_loss += loss.item()
                
                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    imodel.eval()
                    # Turn off gradients to speed up this part
                    with torch.no_grad():
                        for inputs, labels in itest_loader:
                            inputs, labels = inputs.to(idevice), labels.to(idevice)
                          
                            logps = imodel.forward(inputs)
                            
                            batch_loss = icriterion(logps, labels)
                            
                            test_loss += batch_loss.item()
                            
                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                            
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Test loss: {test_loss/len(itest_loader):.3f}.. "
                          f"Test accuracy: {accuracy/len(itest_loader):.3f}")
                    running_loss = 0
                    imodel.train()
                    
        return imodel
    
    except BaseException as exception:
        print("build_and_train_classifier:", repr(exception));
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_type, exc_tb.tb_lineno)
        sys.exit()

#
# Test Network
#
def test_network(ivalidloader, idevice, imodel):

    try:
        
        # Do validation on the test set
        accuracy = 0
        with torch.no_grad():
            imodel.eval()
            for inputs, labels in ivalidloader:
                inputs, labels = inputs.to(idevice), labels.to(idevice)
                #logps = model.forward(inputs)
                logps = imodel(inputs)
        
                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                            
        print(f"Validation accuracy: {accuracy/len(ivalidloader):.3f}")
   
    
    except BaseException as exception:
        print("test_network:", repr(exception));
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_type, exc_tb.tb_lineno)
        sys.exit()

#
# Save Checkpoint
#

def save_checkpoint(imodel, ioptimizer, itest_data, itrain_data, ihidden_units, iarch, iepochs): 

    try:
        #Save the checkpoint 
        imodel.class_to_idx = itrain_data.class_to_idx
        checkpoint= {
                     'output_size':102,
                     'hidden_layer_size':4096,
                     'state_dict':imodel.state_dict(),
                     'class_to_idx':imodel.class_to_idx,
                     'optimizer_state_dict':ioptimizer.state_dict(),
                     'epochs':iepochs,
                     'learningrate':0.003,
                     'dropout':0.2}
        torch.save(checkpoint, 'checkpoint.pth')
        
    except BaseException as exception:
        print("save_checkpoint:", repr(exception));
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_type, exc_tb.tb_lineno)
        sys.exit()  

    return             
#
# Main Routine
#
def main():
                            
    try:
        args = get_input_args()
        
        train_data, valid_data, test_data, trainloader, validloader, testloader = load_the_data()
        
        cat_to_name = label_mapping(args.cat_to_name)
        
        device = torch.device('cuda' if torch.cuda.is_available() and args.gpu==True else 'cpu')
        if(debugFlag):
            print("Device:", device)
            print('Starting build_classifier')
            
        model, optimizer, criterion, inp_features = \
            build_classifier(args.arch, device, args.learning_rate, args.hidden_units)
            
        if(debugFlag):
            print('Ending build_classifier, Starting train_classifier')
            
        model = train_classifier(model, trainloader, \
                         validloader, testloader, optimizer, \
                             criterion, args.epochs, device)
            
        if(debugFlag):
            print('Ending train_classifier, Starting test_network')
    
        test_network(validloader, device, model)
        
        if(debugFlag):
            print('Ending test_network')
            
        save_checkpoint(model, optimizer, test_data, train_data, \
                        args.hidden_units, args.arch, args.epochs)
                            
    except BaseException as exception:
        print("main !", repr(exception));
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_type, exc_tb.tb_lineno)
        sys.exit()                            
    
#
# Main Entry
#
                            
if __name__ == "__main__":
    main()   
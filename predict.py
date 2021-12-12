import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import sys
import argparse
from PIL import Image
import numpy as np
import json

'''
Input args between predict and train
--save_dir
--arch
--learning_rate
--hidden_units
--epochs
--gpu
--top_k
--category_names
'''

debugFlag = True

def get_input_args():
    
    try:
        
        parser = argparse.ArgumentParser(description='Input args')
        
        parser.add_argument('path_to_image', type=str, action='store', 
                            default='flowers/test/1/image_06743.jpg',
                    help='Input path to the image')
        
        parser.add_argument('--save_dir', type=str, action='store', default='.',
                    help='Input Model save dir, default is current dir')
    
        parser.add_argument('--arch', type=str, default="vgg19",
                    help="Input model arch: vgg19, resnet18, densenet121")
    
    
        parser.add_argument('--gpu', default=True,
                    help='use GPU or CPU to train model: True = GPU, False = CPU')
    
        parser.add_argument('--top_k', default=5, type=int, 
                    help="Number of top predictions/ probabilities")                

    
        parser.add_argument('--cat_to_name', action='store', default = 'cat_to_name.json',
                    help='Enter path to image.')

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
# Image processing
#

def process_image(iimage):
    
    try:  
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        
        # Process a PIL image for use in a PyTorch model
        # Using variables as per the documentation
        
        try:
            
            img = Image.open(iimage)
            
        except BaseException as exception:
            print('process_image: Error reading image in process_image - ', repr(exception));
            return ''
    
        # Resize image as per - First, resize the images where the shortest side is 256 pixels, 
        # keeping the aspect ratio. This can be done with the thumbnail or resize methods. 
        # Then you'll need to crop out the center 224x224 portion of the image.
        max_size = (256, 256)
        img.thumbnail(max_size)
        
        # Get new image dimensions
        width, height = img.size
        if(debugFlag):
            print(f'process_image: width: {width}, height: {height}')
        
        new_width = 224
        new_height = 224
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
        
        if(debugFlag):
            print(f'process_image: top: {top}, left: {left}, right: {right}, bottom: {bottom}')
        
        # Crop centre of image
        crop_img = img.crop((left, top, right, bottom))
        
        # Get colour channels
        # Color channels of images are typically encoded as integers 0-255, but the model 
        # expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy 
        # array, which you can get from a PIL image like so np_image = np.array(pil_image).
        np_img = np.array(crop_img)
        
        # Reorder colour channels
        np_img = np_img.transpose((2, 0, 1))
        
        np_img = np_img / 255
        
        # Normalise colours as per For the means, it's [0.485, 0.456, 0.406] and for the standard 
        # deviations [0.229, 0.224, 0.225]. You'll want to subtract the means from each color 
        # channel, then divide by the standard deviation. 
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        if(debugFlag):
            print(f'process_image: mean: {mean}, std: {std}, np_img shape: {np_img.shape}')
        
        #np_img = (np_img - mean) / std
        np_img[0] = (np_img[0] - 0.485)/0.229
        np_img[1] = (np_img[1] - 0.456)/0.224
        np_img[2] = (np_img[2] - 0.406)/0.225
        
        # Convert to torch tensor
        t_img = torch.from_numpy(np_img)
        t_img = t_img.float()
        
        #print(np_img.shape)
        return t_img        
        
    except BaseException as exception:
        print("process_image !", repr(exception));
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_type, exc_tb.tb_lineno)
        sys.exit()

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    try:
        if ax is None:
            fig, ax = plt.subplots()
        
        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.numpy().transpose((1, 2, 0))
        
        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        
        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)
        
        ax.imshow(image)
        
        return ax 

    except BaseException as exception:
        print("imshow !", repr(exception));
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_type, exc_tb.tb_lineno)
        sys.exit()

#
# Pridict image
#
def predict(iimage_path, imodel, icat_to_name, itopk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    try:
        # Implement the code to predict the class from an image file
        # Can we do this using gpu?
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        imodel.to(device)
        
        #Let us load the image
        try:
            imodel.eval()
            
            img = process_image(iimage_path)
            img = img.unsqueeze(0)
            img = img.float()
            
        except BaseException as exception:
            print('predict: Error reading image in predict - ', repr(exception));
            return '', ''
    
        with torch.no_grad():
            logps = imodel(img)
            
        ps = torch.exp(logps)
        top_p, top_class = torch.topk(ps, dim=1, k=itopk)
        
        if(debugFlag):
            print(f'predict: Top Prob Shape: {top_p.shape}, Top Class shape: {top_class.shape}')
            print(f'predict: Top Prob: {top_p}, Top Class: {top_class}')
        
        # Check the class using the top_class
        indicies = np.array(top_class)
        if(debugFlag):
            print(f'predict: indicies: {indicies}')
        index_to_class = {val: key for key, val in imodel.class_to_idx.items()}
        if(debugFlag):
            print(f'predict: index_to_class: {index_to_class}')
        top_classes = [index_to_class[each] for each in indicies[0]]
    
        #print(f'idx2class: {index_to_class}, Top Class: {top_classes}')
            
        class_names = []
        
        for class_idx in top_classes:
                class_names.append(icat_to_name[str(class_idx)])
                
        return top_p.cpu().numpy()[0], class_names

    except BaseException as exception:
        print("predict !", repr(exception));
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_type, exc_tb.tb_lineno)
        sys.exit()

#
# Load checkpoint model file
#
# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(ifilepath, iarch):
    
    try:
        checkpoint = torch.load(ifilepath + '/checkpoint.pth')
        
        model =  getattr(models,iarch)(pretrained=True)
        inp_features = model.classifier[0].in_features
        
        if(debugFlag):
            print(f'load_checkpoint: model: {model}')
            
        for param in model.parameters():
            param.requires_grad=False
        
        model.classifier = nn.Sequential(nn.Linear(inp_features,checkpoint['hidden_layer_size']),
                                     nn.ReLU(),
                                     nn.Dropout(checkpoint['dropout']),
                                     nn.Linear(checkpoint['hidden_layer_size'],checkpoint['output_size']),
                                     nn.LogSoftmax(dim=1))
        
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])
        
        if(debugFlag):
            print(f'load_checkpoint: model: {model}')
        
        return model

    except BaseException as exception:
        print("load_checkpoint !", repr(exception));
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
# Main Routine
#
def main():
                            
    try:
        args = get_input_args()
        
        cat_to_name = label_mapping(args.cat_to_name)
        
        if(debugFlag):
            print('Starting load_checkpoint')
            
        model = load_checkpoint(args.save_dir, args.arch)
                
        if(debugFlag):
            print('Ending load_checkpoint')
                
        device = torch.device('cuda' if torch.cuda.is_available() and args.gpu==True else 'cpu')
        if(debugFlag):
            print("Device:", device)
            print('Starting predict')
            
        probs, classes = predict(args.path_to_image, model, cat_to_name, args.top_k)
        
        if(debugFlag):
            print('Ending predict')
            
        print(f'Top Probability: {probs}')
        print(f'Top Classes: {classes}')
                            
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
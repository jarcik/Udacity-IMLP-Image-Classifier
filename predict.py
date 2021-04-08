import argparse
import json
import torch
import numpy as np
from train import get_model, get_transforms
from torchvision import models
from PIL import Image

#arguments from console
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type = str)
    parser.add_argument("checkpoint", type = str)
    parser.add_argument("--top_k", type = int, default = 1)
    parser.add_argument("--category_names", type = str, default = "cat_to_name.json")
    parser.add_argument("--gpu", type = bool, default=False)
    return parser.parse_args()

#load model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = get_model(checkpoint['name'], checkpoint['hidden_units'])
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
        
    return model

#process image
def process_image(image):
    img = Image.open(image)
    
    transform = get_transforms(True)    
    new_image = transform(img)
    return np.array(new_image)

#predict probs
def predict(image_path, model, category_names, top_k, device):
    model.to(device)
    model.eval()
    
    image = process_image(image_path)    
    image_to_model = torch.from_numpy(image).type(torch.FloatTensor).unsqueeze(0)
    image_to_model.to(device)
    
    with torch.no_grad ():
        prob = model.forward(image_to_model)
        
    model_prob = torch.exp(prob)    
    top_probs, top_labels = model_prob.topk(top_k)
    top_probs = top_probs.numpy()[0]
    top_labels = top_labels.numpy()[0]
    
    cat_to_name = None
    with open(category_names, 'r') as f:
        	cat_to_name = json.load(f)    
    labels = [cat_to_name[str(i)] for i in top_labels]
    
    return top_probs, labels

#show what was predicted
def show_probs(probs, labels):
    for i in range(len(probs)):
        print(f"{i+1}. {labels[i]} with probability {100.0 * probs[i]:.3f}%")

#main
def main():
    args = arg_parse()
    model = load_checkpoint(args.checkpoint)
    device = torch.device("cuda:0" if args.gpu else "cpu")
    probs, labels = predict(args.input, model, args.category_names, args.top_k, device)
    show_probs(probs, labels)
    
if __name__ == "__main__":
    main()
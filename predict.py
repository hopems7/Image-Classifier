#imports here

import argparse
import torch
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import json
print("Imports completed")

def parse_args():
    print("Parsing args...")
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint', metavar='checkpoint', action='store', default='checkpoint.pth')
    #parser.add_argument('--gpu', action='store', default='gpu')
    parser.add_argument('--top_k', dest='top_k', default='3')
    #removed the default from above with the flower filepath
    parser.add_argument('--filepath', dest='filepath', default='flowers/test/10/image_07090.jpg')#'flowers/test/10/image_07090.jpg')
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')
    return parser.parse_args()

def process_image(image):
    #process image for pytorch
    print("Processing image..")
    #convert to PIL image                    
    pil_image=Image.open(image)
                        
    #transforms all in one batch - keep it concise
    image_transforms=transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    pil_image=image_transforms(pil_image)
    np_image=np.array(pil_image) 
    
    
    return np_image
                        
def load_checkpoint(filepath):
    print("Loading from checkpoint..")
    #might be missing some??                    
    checkpoint=torch.load(filepath)
    #model=checkpoint['pretrained_model']
    #changed the bottom
    model=getattr(torchvision.models, checkpoint['pretrained_model'])(pretrained=True)
    model.classifier=checkpoint['classifier']                    
    learning_rate=checkpoint['learning_rate']
    optimizer=checkpoint['optimizer']  
    
    epochs=checkpoint['epochs']
    model.class_to_idx=checkpoint['class_to_idx']                    
    model.load_state_dict(checkpoint['state_dict'])   
                        
    return model                    
                        
def predict(image_path, model, topk=3, gpu='gpu'):
    # Implement the code to predict the class from an image file
    print("In predict function")
    
    #send to correct device
    if gpu == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)
    
    image=process_image(image_path)
    
    torch_image=torch.from_numpy(image)
    torch_image=torch_image.unsqueeze_(0)
    torch_image=torch_image.float()
    model.eval()
    
       
    if gpu =='gpu':
        with torch.no_grad():
            print("Using Cuda")
            output=model.forward(torch_image.cuda())
    else:
        with torch.no_grad():
            print("Using CPU")
            output=model.forward(torch_image.cpu())
  
                        
    ps=F.softmax(output.data, dim=1)
        
    probability=np.array(ps.topk(topk)[0][0])
        
    #the reverse to index to class based on class to index
    idx_to_class={value: key for key, value in model.class_to_idx.items()}
    
    #top classes 
    t_classes=[np.int(idx_to_class[item]) for item in np.array(ps.topk(topk)[1][0])]
    
    return probability, t_classes
                        
def load_image_names(filename):
    print("Loading images from path..")
    with open(filename) as f:
         cat_names=json.load(f)
    return cat_names
                        
                        
def main():
    print("In main...")
    args = parse_args()
    top_k=args.top_k                    
    checkpoint=args.checkpoint                    
    model=load_checkpoint(checkpoint)
    cat_to_name=load_image_names(args.category_names)
                        
    img_pth=args.filepath
    t_probs, classes = predict(img_pth, model, int(args.top_k), args.gpu)                
    image_names=[cat_to_name[str(index)] for index in classes]                    
    
    print("Results for image: ", img_pth)
    print(image_names, t_probs)
    
    print("For top_k classes and probabilities")
    for i in range(len(image_names)):
        print("{} Probability: {}".format(image_names[i], t_probs[i]))               
        i+=1  
    
    #other test cases: 'flowers/test/1/image_06743.jpg', 
# 'flowers/test/10/image_07090.jpg'  
#'flowers/test/85/image_04805.jpg'
                        
   
if __name__ == '__main__':
    main()                 
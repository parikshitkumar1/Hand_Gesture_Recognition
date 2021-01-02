from torchvision import models, transforms
import torch
from PIL import Image
import cv2



def predict(img):
    model = torch.load("WEIGHTS.pth", map_location = "cpu")
    


    im_pil = Image.fromarray(img)
    transform = (transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()]))
    batch_t = torch.unsqueeze(transform(im_pil), 0)

    model.eval()
    out = model(batch_t)

    with open('image_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx]) for idx in indices[0][:1]]
    
    
    
    
'''
0, none
1, paper
2, rock
3, scissors

'''

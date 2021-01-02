from PIL import Image
from model import predict
import torch
import cv2
import numpy as np
from random import choice

###############################################

import torch.nn as nn
import torch.nn.functional as F

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) #  loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
            
            


class gestureModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   #128

            nn.Conv2d(16,32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #64

            nn.Conv2d(32,32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #32

            nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #16

            nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #8

            nn.Flatten(), 
            nn.Linear(128*8*8,1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4))
        
    def forward(self, xb):
        return self.network(xb)
        


###################################################

REV_CLASS_MAP = {   #mapping predicted code to gesture
    '1': "ONE",
    '10': "TEN",
    '11': "NONE",
    '2': "TWO",
    '3': "THREE",
    '4': "FOUR",
    '5': "FIVE",
    '6': "SIX",
    '7': "SEVEN",
    '8': "EIGHT",
    '9': "NINE"
}


model = torch.load("WEIGHTS.pth", map_location = "cpu")  #loading weights 
model.eval()
cap = cv2.VideoCapture(0)

prev_move = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)
    roi = frame[100:500, 100:500]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    

    # sends move for mapping
    pred = predict(img)
    
    
    user_move_name = REV_CLASS_MAP[pred[0]]


    # displays gesture information
    font = cv2.FONT_ITALIC
    cv2.putText(frame, "Your Gesture is: " + user_move_name,
                (50, 50), font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    

    

    cv2.imshow("HAND GESTURE RECOGNITION --- PRESS 'Q' TO EXIT", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

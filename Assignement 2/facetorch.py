import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from keras import layers,models
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

#SECTION - Constant
# --------------------------------- CONSTANT --------------------------------- #

encoding_dim = 32
BATCH_SIZE = 32
EPOCHS  = 25

#NOTE - 320*64*64 / 80*64*64
#SECTION - Retrieve Data
# ------------------------------- RETRIEVE DATA ------------------------------ #
data = fetch_olivetti_faces()
faces = data.images
train, test = train_test_split(faces,test_size=0.2)

train = torch.tensor(train.reshape(len(train),1,64,64))
test = torch.tensor(test.reshape(len(test),1,64,64))

#SECTION - Model Creation
# ------------------------------ MODEL CREATION ------------------------------ #
class AE(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.Econv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.Epool = nn.MaxPool2d(2,2)
        self.Econv2 = nn.Conv2d(16, 64, 3, padding=1)

        self.Dconv1 = nn.Conv2d(64, 16, 3, padding=1)
        self.Dup = nn.Upsample(scale_factor=2)
        self.Dconv2 = nn.Conv2d(16, 1, 3, padding=1)
         
        
    def forward(self, x):
        #Encoder
        x = self.Econv1(x)
        x = self.relu(x)
        x = self.Epool(x)
        x = self.Econv2(x)
        x = self.relu(x)
        x = self.Epool(x)

        #Decoder
        x = self.Dconv1(x)
        x = self.relu(x)
        x = x.reshape(1,16,16,16)
        x = self.Dup(x)
        x = self.Dconv2(x)
        x = self.relu(x)
        x = self.Dup(x)
        x = self.sigmoid(x)
        x = x.reshape(1,64,64)
        return x

autoencoder = AE()
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

#SECTION - Training
# --------------------------------- TRAINING --------------------------------- #
for epoch in range(EPOCHS):
    sum_loss = 0
    for images in train:
        optimizer.zero_grad()
        outputs = autoencoder(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        sum_loss+=loss.item()

    print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {sum_loss:.4f}')
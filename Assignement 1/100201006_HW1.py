# ---------------------------------- IMPORT ---------------------------------- #
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist

# ---------------------------------------------------------------------------- #
#                                BOSTON HOUSING                                #
# ---------------------------------------------------------------------------- #


# --------------------------------- CONSTANT --------------------------------- #
PATH = "./introduction_deep/Assignement 1/"
EPOCHS = 20
x = np.linspace(0,EPOCHS-1,EPOCHS)

# ------------------------------ DATASET LOADING ----------------------------- #  
data = pd.read_csv(PATH+"TheBostonHousingDataset.csv").values
data = train_test_split(data,train_size=int(80*len(data)/100))
trainloader = torch.tensor(data[0])
testloader = torch.tensor(data[1])

# -------------------------- NEURAL NETWORK CREATION ------------------------- #
class BostonFFN(nn.Module):
    def __init__(self):
        super(BostonFFN, self).__init__()
        self.fc1 = nn.Linear(13, 6).to(torch.float64)
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Linear(6, 1).to(torch.float64)
    
    def forward(self, inp):
        out = self.fc1(inp)
        out = self.ReLU(out)
        out = self.fc2(out)
        out = self.ReLU(out)
        return out

boston_model = BostonFFN()
criterion = nn.MSELoss()
optimizer = optim.Adam(boston_model.parameters(), lr=0.01)

# --------------------------------- TRAINING --------------------------------- #
y_boston_loss = []
y_boston_diff = []
for epoch in range(0): 
    sum_loss = 0
    sum_difference = 0
    for data in trainloader:
        inputs = data[:-1]
        labels = data[-1]
        optimizer.zero_grad()
        outputs = boston_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        sum_loss+= loss.item()
        sum_difference+= abs(labels-outputs.item())
        optimizer.step()
    y_boston_loss.append(sum_loss/len(trainloader))
    y_boston_diff.append(sum_difference/len(trainloader))
    print(f"[EPOCH {epoch}/{EPOCHS}] Average loss: {sum_loss/len(trainloader)}, Average difference: {sum_difference/len(trainloader)}")
# plt.plot(x, y_boston_diff)
# plt.plot(x, y_boston_loss)
# plt.savefig(PATH+"boston.png")

# -------------------------------- EVALUATION -------------------------------- #
with torch.no_grad():
    sum_loss = 0
    sum_difference = 0
    for data in testloader:
        inputs = data[:-1]
        labels = data[-1]
        outputs = boston_model(inputs)
        loss = criterion(outputs, labels)
        sum_loss+= loss.item()
        sum_difference+= abs(labels-outputs.item())
    print(f"[VALIDATION] Average loss: {sum_loss/len(testloader)}, Average difference: {sum_difference/len(testloader)}")


# ---------------------------------------------------------------------------- #
#                                 BREAST CANCER                                #
# ---------------------------------------------------------------------------- #


# ---------------------------- DATA PREPROCESSING ---------------------------- #
data_task2 = load_breast_cancer()
x_train,x_test,y_train,y_test=train_test_split(data_task2.data,data_task2.target,test_size=0.2,random_state=5)
x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train).unsqueeze(1).to(torch.float64)
x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test)

class BinaryCancer(nn.Module):
    def __init__(self):
        super(BinaryCancer, self).__init__()
        self.fc = nn.Linear(30, 15).to(torch.float64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(15, 1).to(torch.float64)
        self.sig = nn.Sigmoid().to(torch.float64)
    
    def forward(self, inp):
        out = self.fc(inp)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sig(out)
        return out
    
cancer_model = BinaryCancer()
criterion_cancer = nn.BCELoss()
optimizer = optim.Adam(cancer_model.parameters(), lr=0.01)

# --------------------------------- TRAINING --------------------------------- #
for epoch in range(0): 
    sum_loss = 0
    for input,label in zip(x_train,y_train):
        optimizer.zero_grad()
        outputs = cancer_model(input)
        loss = criterion_cancer(outputs, label)
        loss.backward()
        sum_loss+= loss.item()
        optimizer.step()
        print(f"Out:{outputs.item()}, Tar: {label.item()}")
    print(f"[EPOCH {epoch}/{EPOCHS}] Average loss: {sum_loss/len(x_train)}")


# ---------------------------------------------------------------------------- #
#                                     MNSIT                                    #
# ---------------------------------------------------------------------------- #


# ---------------------------- DATA PREPROCESSING ---------------------------- #
(train_images,train_labels),(test_images, test_labels) = fashion_mnist.load_data()

train_images, test_images = train_images/255,test_images/255

mnist_x_train = torch.tensor(train_images).to(torch.float32)
mnist_y_train = torch.tensor(train_labels).to(torch.float32)
mnist_x_test = torch.tensor(test_images).to(torch.float32)
mnist_y_test = torch.tensor(test_labels).to(torch.float32)


# ----------------------- NEURAL NETWORK INITIALIZATION ---------------------- #
class MnistClassifier(nn.Module):
    def __init__(self):
        super(MnistClassifier, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.maxp = nn.MaxPool2d(2)
        self.smax = nn.Softmax()
    
    def forward(self, inp):
        out = self.conv1(inp)
        out = self.maxp(out)
        out = self.conv2(out)
        out = self.maxp(out)
        out = self.relu(out)
        out = self.smax(out)
        return out

mnist_model = MnistClassifier()
criterion_mnist = nn.CrossEntropyLoss()
optimizer = optim.Adam(cancer_model.parameters(), lr=0.01)

for epoch in range(EPOCHS):
    sum_loss = 0
    for items, labels in zip(mnist_x_train, mnist_y_train):
        items = items[None,:,:]
        optimizer.zero_grad()
        outputs = mnist_model(items)
        loss = criterion(outputs, labels)
        loss.backward()
        sum_loss+= loss.item()
        optimizer.step()
    print(f"[EPOCH {epoch}/{EPOCHS}] Average loss: {sum_loss/len(mnist_x_train)}")

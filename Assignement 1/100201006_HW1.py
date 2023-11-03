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
from torchmetrics.classification import BinaryAccuracy,Accuracy
from torch.utils.data import DataLoader,TensorDataset

# ---------------------------------------------------------------------------- #
#                                BOSTON HOUSING                                #
# ---------------------------------------------------------------------------- #

print("# ------------------------------ BOSTON HOUSING ------------------------------ #")

# --------------------------------- CONSTANT --------------------------------- #
PATH = "./introduction_deep/Assignement 1/"
EPOCHS = 20
BATCH_SIZE = 64
x = np.linspace(0,EPOCHS-1,EPOCHS)
x_validation = np.linspace(0,EPOCHS-1,int(EPOCHS/5)+1)
x_validation = x_validation[1:]


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
        return out

boston_model = BostonFFN()
criterion = nn.MSELoss()
optimizer = optim.Adam(boston_model.parameters(), lr=0.01)


# --------------------------------- TRAINING --------------------------------- #
y_boston_loss = []
y_boston_diff = []
y_boston_val_loss = []
for epoch in range(EPOCHS): 
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
    print(f"[EPOCH {epoch+1}/{EPOCHS}] Average loss: {sum_loss/len(trainloader)}, Average difference: {sum_difference/len(trainloader)}")
    
    
    # -------------------------------- VALIDATION -------------------------------- #
    if (epoch+1)%5==0:
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
        y_boston_val_loss.append(sum_loss/len(testloader))
plt.plot(x, y_boston_diff)
plt.plot(x, y_boston_loss)
plt.plot(x_validation,y_boston_val_loss)
plt.savefig(PATH+"boston.png")   


# ---------------------------------------------------------------------------- #
#                                 BREAST CANCER                                #
# ---------------------------------------------------------------------------- #

print("# ------------------------------- BREAST CANCER ------------------------------ #")

# ---------------------------- DATA PREPROCESSING ---------------------------- #
data_task2 = load_breast_cancer()
x_train,x_test,y_train,y_test=train_test_split(data_task2.data,data_task2.target,test_size=0.2,random_state=5)
x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train).unsqueeze(1).to(torch.float64)
x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test).unsqueeze(1).to(torch.float64)

cancer_train_dataset = TensorDataset(x_train,y_train)
cancer_train_loader = DataLoader(cancer_train_dataset,batch_size=BATCH_SIZE,shuffle=True)
cancer_test_dataset = TensorDataset(x_test,y_test)
cancer_test_loader = DataLoader(cancer_test_dataset,batch_size=BATCH_SIZE,shuffle=True)


# ------------------------------ MODEL CREATION ------------------------------ #
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
metric = BinaryAccuracy()

# --------------------------------- TRAINING --------------------------------- #
y_cancer_loss = []
y_cancer_acc = []
y_cancer_val_loss = []
for epoch in range(EPOCHS): 
    sum_loss = 0
    accuracy = 0
    for input,label in cancer_train_loader:
        optimizer.zero_grad()
        outputs = cancer_model(input)
        loss = criterion_cancer(outputs, label)
        loss.backward()
        sum_loss+= loss.item()
        optimizer.step()
        accuracy+=metric(outputs,label)
    print(f"[EPOCH {epoch+1}/{EPOCHS}] Average loss: {sum_loss*BATCH_SIZE/len(cancer_train_loader)}, Accuracy: {accuracy*BATCH_SIZE/len(cancer_train_loader)}%")
    y_cancer_loss.append(sum_loss*BATCH_SIZE/len(cancer_train_loader))
    y_cancer_acc.append(accuracy*BATCH_SIZE/len(cancer_train_loader))
    
    # -------------------------------- VALIDATION -------------------------------- #
    if (epoch+1)%5==0:
        sum_loss = 0
        for input,label in cancer_test_loader:
            with torch.no_grad():
                outputs = cancer_model(input)
                loss = criterion_cancer(outputs, label)
                sum_loss+= loss.item()
        print(f"[VALIDATION] Average loss: {sum_loss*BATCH_SIZE/len(cancer_test_loader)}")
        y_cancer_val_loss.append(sum_loss*BATCH_SIZE/len(cancer_test_loader))

plt.clf()
plt.plot(x,y_cancer_loss)
plt.plot(x,y_cancer_acc)
plt.plot(x_validation,y_cancer_val_loss)
plt.savefig(PATH+"cancer.png")

# ---------------------------------------------------------------------------- #
#                                     MNIST                                    #
# ---------------------------------------------------------------------------- #

print("# ----------------------------------- MNIST ---------------------------------- #")

# ---------------------------- DATA PREPROCESSING ---------------------------- #
(train_images,train_labels),(test_images, test_labels) = fashion_mnist.load_data()

train_images, test_images = train_images/255,test_images/255

mnist_x_train = torch.tensor(train_images).to(torch.float64)
train_labels = [np.eye(1, 10, k = train_labels[i]) for i in train_labels]
mnist_y_train = torch.tensor(train_labels).to(torch.float64)
mnist_x_test = torch.tensor(test_images).to(torch.float64)
test_labels = [np.eye(1, 10, k = test_labels[i]) for i in test_labels]
mnist_y_test = torch.tensor(test_labels).to(torch.float64)

mnist_train_dataset = TensorDataset(mnist_x_train,mnist_y_train)
mnist_train_loader = DataLoader(mnist_train_dataset,batch_size=BATCH_SIZE,shuffle=True)
mnist_test_dataset = TensorDataset(mnist_x_test,mnist_y_test)
mnist_test_loader = DataLoader(mnist_test_dataset,batch_size=BATCH_SIZE,shuffle=True)


# ----------------------- NEURAL NETWORK INITIALIZATION ---------------------- #
class MnistClassifier(nn.Module):
    def __init__(self):
        super(MnistClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5).to(torch.float64)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=5).to(torch.float64)
        self.maxpool = nn.MaxPool2d(2).to(torch.float64)
        self.fc1 = nn.Linear(16 * 4 * 4, 128).to(torch.float64)
        self.fc2 = nn.Linear(128, 10).to(torch.float64)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        return out

mnist_model = MnistClassifier()
criterion_mnist = nn.CrossEntropyLoss()
optimizer = optim.Adam(mnist_model.parameters(), lr=0.01)
metric = Accuracy(task="multiclass", num_classes=10)


# --------------------------------- TRAINING --------------------------------- #
y_mnist_loss = []
y_mnist_acc = []
y_mnist_val_loss = []
for epoch in range(EPOCHS):
    sum_loss = 0
    accuracy = 0
    for items, labels in mnist_train_loader:
        items = items[:,None,:,:]
        optimizer.zero_grad()
        outputs = mnist_model(items)
        outputs = outputs.view(-1,1,10)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        accuracy += metric(outputs,labels)
    print(f"[EPOCH {epoch+1}/{EPOCHS}] Average loss: {sum_loss*BATCH_SIZE/len(mnist_train_loader)},Accuracy: {accuracy*BATCH_SIZE/len(mnist_test_loader)}%")
    y_mnist_loss.append(sum_loss*BATCH_SIZE/len(mnist_train_loader))
    y_mnist_acc.append(accuracy*BATCH_SIZE/len(mnist_test_loader))
    
    # -------------------------------- VALIDATION -------------------------------- #
    if (epoch+1)%5==0:
        sum_loss = 0
        accuracy = 0
        for items, labels in mnist_test_loader:
            with torch.no_grad():
                items = items[:,None,:,:]
                outputs = mnist_model(items)
                outputs = outputs.view(-1,1,10)
                loss = criterion(outputs, labels)
                sum_loss += loss.item()
                
        print(f"[VALIDATION] Average loss: {sum_loss*BATCH_SIZE/len(mnist_train_loader)}")
        y_cancer_val_loss.append(sum_loss*BATCH_SIZE/len(mnist_train_loader))

plt.clf()
plt.plot(x,y_mnist_loss)
plt.plot(x,y_mnist_acc)
plt.plot(x_validation,y_mnist_val_loss)
plt.savefig(PATH+"mnist.png")
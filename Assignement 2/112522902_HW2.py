# ---------------------------------------------------------------------------- #
#                                HUMAN ACTIVITY                                #
# ---------------------------------------------------------------------------- #

import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader,TensorDataset

# --------------------------------- CONSTANTS -------------------------------- #

BATCH_SIZE = 128
learning_rate = 0.01
hidden_size = 128
EPOCHS = 20

x = np.linspace(0,EPOCHS-1,EPOCHS)
x_validation = np.linspace(0,EPOCHS-1,int(EPOCHS/5)+1)
x_validation = x_validation[1:]

# ------------------------------ DATA RETRIEVING ----------------------------- #

with open("./datasets/UCI HAR Dataset/UCI HAR Dataset/test/X_test.txt","r") as f:
    test_item = f.readlines()

with open("./datasets/UCI HAR Dataset/UCI HAR Dataset/test/y_test.txt","r") as f:
    test_labels = f.readlines()

with open("./datasets/UCI HAR Dataset/UCI HAR Dataset/test/subject_test.txt","r") as f:
    test_subjects = f.readlines()

with open("./datasets/UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt","r") as f:
    train_item = f.readlines()

with open("./datasets/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt","r") as f:
    train_labels = f.readlines()

with open("./datasets/UCI HAR Dataset/UCI HAR Dataset/train/subject_train.txt","r") as f:
    train_subjects = f.readlines()

with open("./datasets/UCI HAR Dataset/UCI HAR Dataset/activity_labels.txt") as f:
    labels_list = f.readlines()


# ---------------------------- DATA PREPROCESSING ---------------------------- #

input_size = len(train_item[0].split())+1

idx2label = {int(key): value.strip() for key, value in (item.split() for item in labels_list)}
label2idx = {key.strip(): int(value) for value, key in (item.split() for item in labels_list)}

def text2array(file):
    array = []
    for line in file:
        current = []
        for figure in line.split():
            current.append(float(figure[:-5]) * 10 ** -int(figure[-1]))
        array.append(current)
    return(array)

train_subjects = list(map(int,train_subjects))
train_item = text2array(train_item)
for line,subject in zip(train_item,train_subjects):
    line.insert(0,(subject-15)/15)
train_labels = [[1 if i == int(num[:-1]) else 0 for i in range(6)] for num in train_labels]

test_subjects = list(map(int,test_subjects))
test_item = text2array(test_item)
for line,subject in zip(test_item,test_subjects):
    line.insert(0,(subject-15)/15)
test_labels = [[1 if i == int(num[:-1]) else 0 for i in range(6)] for num in test_labels]

plt.hist(train_labels+test_labels)
plt.show()

train_dataset = TensorDataset(torch.tensor(train_item),torch.tensor(train_labels))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = TensorDataset(torch.tensor(test_item),torch.tensor(test_labels))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ------------------------------- LSTM CREATION ------------------------------ #
class LSTM(nn.Module):
    def __init__(self, input_size,hidden_size,output_size):
        super(LSTM,self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
 
        self.fc1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(int(hidden_size/2),output_size)
        
    
    def forward(self,x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).requires_grad_()
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).requires_grad_()

        x = x.unsqueeze(1)

        x,(h0,c0) = self.lstm(x,(h0,c0))
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x
    
model = LSTM(input_size,hidden_size,len(labels_list))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

# --------------------------------- TRAINING --------------------------------- #
y_loss = []
y_acc = []
y_val_loss = []
y_val_acc = []

for epoch in range(EPOCHS):
    sum_loss = 0
    total = 0
    TP = 0
    for batch, (items,label) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(items)
        outputs = outputs.view(-1,6)
        loss = criterion(outputs,label.to(torch.float))
        for output,answer in zip(outputs,label):
            total+=1
            if(torch.argmax(output) == torch.argmax(answer)):
                TP+=1   
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    
    y_loss.append(sum_loss/len(train_loader))
    y_acc.append(TP*100/total)
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {sum_loss/len(train_loader)}, Accuracy: {TP*100/total}")

# ----------------------------------- TEST ----------------------------------- #

    if (epoch+1)%5 == 0:
        TP=0
        total = 0
        for batch, (items,label) in enumerate(test_loader):
            with torch.no_grad():
                outputs = model(items)
                outputs = outputs.view(-1,6)
                loss = criterion(outputs,label.to(torch.float))
                for output,answer in zip(outputs,label):
                    total+=1
                    if(torch.argmax(output) == torch.argmax(answer)):
                        TP+=1  
                sum_loss += loss.item()

        y_val_loss.append(sum_loss/len(test_loader))
        y_val_acc.append(TP*100/total)
        print(f"VALIDATION, Loss: {sum_loss/len(test_loader)}, Accuracy: {TP*100/total}")


# --------------------------------- PLOTTING --------------------------------- #
plt.plot(x,y_loss,label = "Loss")
plt.plot(x,y_acc, label = "Accuracy")
plt.plot(x_validation,y_val_loss, label = "Validation loss")
plt.plot(x_validation,y_val_acc, label = "Validation accuracy")
plt.legend()
plt.savefig("./human_activity.png")



# ---------------------------------------------------------------------------- #
#                             TEMPERATURE FORECAST                             #
# ---------------------------------------------------------------------------- #
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --------------------------------- CONSTANT --------------------------------- #
hidden_size = 16
learning_rate = 0.01
EPOCHS = 20

x = np.linspace(0,EPOCHS-1,EPOCHS)
x_validation = np.linspace(0,EPOCHS-1,int(EPOCHS/5)+1)
x_validation = x_validation[1:]

# ---------------------------- DATA PREPROCESSING ---------------------------- #
with open("./Assignement 2/daily-minimum-temperatures-in-me.csv") as f:
    file = f.readlines()

file = file[1:-3]

items = []
labels = []
for line in file:
    splitted = line.split(",")
    try:
        labels.append(float(splitted[1]))
    except:
        continue
    current = splitted[0]
    current = current[1:-1]
    current = current.split("-")

    current = [int(item) for item in current]
    current.append(current[2]/31)
    current[0] = (current[0] - 1980)/10
    current[2] = np.sin(2 * np.pi * current[1] / 12)
    current[1] = np.cos(2 * np.pi * current[1] / 12)
    input_size = len(current)
    items.append([current])

plt.boxplot(labels)
plt.show()

train_items, test_items, train_labels, test_labels = train_test_split(items, labels, test_size=0.2,shuffle=True)

items = torch.tensor(train_items).to(torch.float32)
labels = torch.tensor(train_labels)

test_items = torch.tensor(test_items).to(torch.float32)
test_labels = torch.tensor(test_labels)

# ------------------------------------ GRU ----------------------------------- #
class GRU(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(GRU,self).__init__()

        self.gru = nn.GRU(input_size,hidden_size)
        self.fc1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(int(hidden_size/2),1)


    def forward(self, x):
        h0 = torch.zeros(1, self.gru.hidden_size)
        x,h0 = self.gru(x,h0)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

model = GRU(input_size,hidden_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

# --------------------------------- TRAINING --------------------------------- #
y_acc = []
y_loss = []
y_val_loss = []
y_val_acc = []

for epoch in range(EPOCHS):
    sum_loss = 0
    acc = 0
    for item,label in zip(items,labels):
        optimizer.zero_grad()
        output = model(item)
        acc += abs(output.item()-label.item())
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    y_loss.append(sum_loss/len(items))
    y_acc.append(acc/len(items))

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {sum_loss/len(items)}, Difference: {acc/len(items)}")


# -------------------------------- VALIDATION -------------------------------- #
    if (epoch+1)%5==0:
        with torch.no_grad():
            sum_loss = 0
            acc = 0
            for item,label in zip(test_items,test_labels):
                output = model(item)
                acc += abs(output.item()-label.item())
                loss = criterion(output,label)
                sum_loss += loss.item()

        y_val_acc.append(acc/len(items))
        y_val_loss.append(sum_loss/len(items))
        print(f"EVALUATION, Loss: {sum_loss/len(items)}, Difference: {acc/len(items)}")

# --------------------------------- PLOTTING --------------------------------- #
plt.plot(x,y_loss,label="Loss")
plt.plot(x,y_acc,label = "Difference")
plt.plot(x_validation,y_val_loss, label = "Validation loss")
plt.plot(x_validation,y_val_acc, label = "Validation difference")
plt.legend()
plt.savefig("./cos_temperature_forecast.png")

# ---------------------------------------------------------------------------- #
#                                     FACES                                    #
# ---------------------------------------------------------------------------- #

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from keras import layers,models
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

# --------------------------------- CONSTANT --------------------------------- #

EPOCHS  = 25

#NOTE - 320*64*64 / 80*64*64
# ------------------------------- RETRIEVE DATA ------------------------------ #
data = fetch_olivetti_faces()
faces = data.images
train, test = train_test_split(faces,test_size=0.2,shuffle=True)

train = torch.tensor(train.reshape(len(train),1,64,64))
test = torch.tensor(test.reshape(len(test),1,64,64))

# ------------------------------ MODEL CREATION ------------------------------ #
class AE(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.Econv1 = nn.Conv2d(1, 16, 3, padding=1)
        nn.init.xavier_uniform_(self.Econv1.weight)
        self.Epool = nn.MaxPool2d(2,2)
        self.Econv2 = nn.Conv2d(16, 64, 3, padding=1)
        nn.init.xavier_uniform_(self.Econv2.weight)

        self.Dconv1 = nn.Conv2d(64, 16, 3, padding=1)
        nn.init.xavier_uniform_(self.Dconv1.weight)
        self.Dup = nn.Upsample(scale_factor=2)
        self.Dconv2 = nn.Conv2d(16, 1, 3, padding=1)
        nn.init.xavier_uniform_(self.Dconv2.weight)
         
        
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

# --------------------------------- TRAINING --------------------------------- #

y_loss = []
y_val_loss = []
last_loss = 0

for epoch in range(EPOCHS):
    sum_loss = 0
    for images in train:
        optimizer.zero_grad()
        outputs = autoencoder(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        sum_loss+=loss.item()
    y_loss.append(sum_loss)
    print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {sum_loss:.4f}')

    if (epoch+1)%5==0:
        with torch.no_grad():
            sum_loss=0
            for images in test:
                outputs = autoencoder(images)
                loss = criterion(outputs,images)
                sum_loss+=loss.item()
            y_val_loss.append(sum_loss)
            y_val_loss.append(sum_loss)
            print(f'VALIDATION, Loss: {sum_loss:.4f}')

    if abs(sum_loss - last_loss)<0.01:
        plt.imshow(outputs.detach().reshape(64, 64))
        plt.show()
        break
    else:
        last_loss = sum_loss

x = np.linspace(0,epoch,epoch+1)
x_validation = [0]*len(y_val_loss)
for i in range(len(x_validation)):
    x_validation[i]+=i*5

plt.clf()
plt.plot(x,y_loss,label="Loss")
plt.plot(x_validation,y_val_loss, label = "Validation loss")
plt.legend()
plt.savefig("./face_graph.png")
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader,TensorDataset

#SECTION - Constants
# --------------------------------- CONSTANTS -------------------------------- #

BATCH_SIZE = 128
learning_rate = 0.01
hidden_size = 128
EPOCHS = 20

x = np.linspace(0,EPOCHS-1,EPOCHS)
x_validation = np.linspace(0,EPOCHS-1,int(EPOCHS/5)+1)
x_validation = x_validation[1:]

#SECTION - Data retrieving
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


#SECTION - Data Preprocessing
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


#SECTION - LSTM
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

#SECTION - Training
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

#SECTION - Evaluate
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


#SECTION - Plotting
# --------------------------------- PLOTTING --------------------------------- #
plt.plot(x,y_loss,label = "Loss")
plt.plot(x,y_acc, label = "Accuracy")
plt.plot(x_validation,y_val_loss, label = "Validation loss")
plt.plot(x_validation,y_val_acc, label = "Validation accuracy")
plt.legend()
plt.savefig("./human_activity.png")
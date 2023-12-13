import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#SECTION - Constant
# --------------------------------- CONSTANT --------------------------------- #
hidden_size = 16
learning_rate = 0.01
EPOCHS = 20

x = np.linspace(0,EPOCHS-1,EPOCHS)
x_validation = np.linspace(0,EPOCHS-1,int(EPOCHS/5)+1)
x_validation = x_validation[1:]

#SECTION - Data prepcrocessing
# ---------------------------- DATA PREPROCESSING ---------------------------- #
with open("./daily-minimum-temperatures-in-me.csv") as f:
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

    #NOTE - Not sure about this normalization may supress it later
    #NOTE - can try to normalize using sin/cos bc of cyclic repetition
    current = [int(item) for item in current]
    current.append(current[2]/31)
    current[0] = (current[0] - 1980)/10
    current[2] = np.sin(2 * np.pi * current[1] / 12)
    current[1] = np.cos(2 * np.pi * current[1] / 12)
    input_size = len(current)
    items.append([current])

train_items, test_items, train_labels, test_labels = train_test_split(items, labels, test_size=0.2)

items = torch.tensor(train_items).to(torch.float32)
labels = torch.tensor(train_labels)

test_items = torch.tensor(test_items).to(torch.float32)
test_labels = torch.tensor(test_labels)

#SECTION - GRU
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

#SECTION - Training
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


#SECTION - Evaluation
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

#SECTION - Plotting
# --------------------------------- PLOTTING --------------------------------- #
plt.plot(x,y_loss,label="Loss")
plt.plot(x,y_acc,label = "Difference")
plt.plot(x_validation,y_val_loss, label = "Validation loss")
plt.plot(x_validation,y_val_acc, label = "Validation difference")
plt.legend()
plt.savefig("./cos_temperature_forecast.png")
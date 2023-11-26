import torch

import torch.nn as nn
import torch.optim as optim

#SECTION - Constant
# --------------------------------- CONSTANT --------------------------------- #
hidden_size = 16
learning_rate = 0.01
EPOCHS = 20


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
    input_size = len(current)
    current[0] = (current[0] - 1980)/10
    current[1] = current[1]/12
    current[2] = current[2]/31
    items.append([current])

items = torch.tensor(items)
labels = torch.tensor(labels)
print(f"{items.shape},{labels.shape}")


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

for epoch in range(EPOCHS):
    sum_loss = 0
    TP=0
    for item,label in zip(items,labels):
        optimizer.zero_grad()
        output = model(item)
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {sum_loss/len(items)}, Accuracy: {TP*100/len(items)}")

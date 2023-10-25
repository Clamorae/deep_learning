# ---------------------------------- IMPORT ---------------------------------- #
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split

import pandas as pd

# ---------------------------------------------------------------------------- #
#                                BOSTON HOUSING                                #
# ---------------------------------------------------------------------------- #


# --------------------------------- CONSTANT --------------------------------- #
PATH = "./introduction_deep/Assignement 1/"

# ------------------------------ DATASET LOADING ----------------------------- #  
data = pd.read_csv(PATH+"TheBostonHousingDataset.csv").values
data = train_test_split(data)
# train = data[0]
# test = data[1]
trainloader = torch.tensor(data[0])
testloader = torch.tensor(data[1])

# -------------------------- NEURAL NETWORK CREATION ------------------------- #
class BostonFFN(nn.Module):
    def __init__(self):
        super(BostonFFN, self).__init__()
        self.fc1 = nn.Linear(13, 6).to(torch.float64)
        self.fc2 = nn.Linear(6, 1).to(torch.float64)
    
    def forward(self, inp):
        out = self.fc1(inp)
        out = self.fc2(out)
        return out

boston_model = BostonFFN()
criterion = nn.MSELoss()
optimizer = optim.Adam(boston_model.parameters(), lr=0.01)

# --------------------------------- TRAINING --------------------------------- #
for epoch in range(20):  # loop over the dataset multiple times
    for data in trainloader:
        # get the inputs; data is a list of [inputs, labels]
        inputs = data[:-1]
        labels = data[-1]
        optimizer.zero_grad()
        outputs = boston_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# -------------------------------- EVALUATION -------------------------------- #
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs = data[:-1]
        labels = data[-1]
        outputs = boston_model(inputs)
        print(f"output = {outputs.item()}, real value = {labels}, diff = {labels-outputs.item()}")




# ---------------------------------------------------------------------------- #
#                                 BREAST CANCER                                #
# ---------------------------------------------------------------------------- #


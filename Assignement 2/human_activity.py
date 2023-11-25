import torch

import numpy as np

from torch.utils.data import DataLoader,TensorDataset

#SECTION - Constants
# --------------------------------- CONSTANTS -------------------------------- #

BATCH_SIZE = 32


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
train_labels = [np.array([1 if i == num else 0 for i in range(6)]) for num in train_labels]

test_subjects = list(map(int,test_subjects))
test_item = text2array(test_item)
for line,subject in zip(test_item,test_subjects):
    line.insert(0,(subject-15)/15)
test_labels = [np.array([1 if i == num else 0 for i in range(6)]) for num in test_labels]

train_dataset = TensorDataset(torch.tensor(train_item),torch.tensor(train_labels))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = TensorDataset(torch.tensor(test_item),torch.tensor(test_labels))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

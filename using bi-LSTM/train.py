import pandas as pd
import numpy as np
import os
import random
import string
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

def train_loop(data_loader, model, loss_fn, optimizer, loss_estimate, batch_no, epoch, epoch_no):
    size = len(data_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()                
        if batch % 1000 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            loss_estimate.append(loss)
            batch_no.append(current)
            epoch_no.append(epoch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(data_loader, model, loss_fn):
    size = len(data_loader.dataset)
    model.eval()
    num_batches = len(data_loader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for (X, y) in data_loader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(0) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

class CustomDatasetTrain(Dataset):
    def __init__(self, X_train, y_train):
        self.features = X_train
        self.label = y_train
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.label[idx]
        sample = {"features": features, "label": label}
        return features, label

class extract_tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor[:, -1, :]


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.LSTM_stack = nn.Sequential(
            nn.Embedding(64, 32, max_norm=1, norm_type=2),
            nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True, dropout=0.2, bidirectional=True),
            extract_tensor(),
            nn.Linear(128, 26)
        )
    
    def forward(self, x):
        logits = self.LSTM_stack(x)
        return logits

def create_dataloader(input_tensor, target_tensor):
    all_features_data = CustomDatasetTrain(input_tensor, target_tensor)
    all_features_dataloader = DataLoader(all_features_data, batch_size=128, shuffle=True)
    return all_features_dataloader

def save_model(model):
    torch.save(model.state_dict(), "bi-lstm-embedding-model-state.pt")    

def train_model(input_tensor, target_tensor):
    all_features_dataloader = create_dataloader(input_tensor, target_tensor)
    model = NeuralNetwork()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_estimate = []
    batch_no = []
    epoch_no = []
    epochs = 8
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(all_features_dataloader, model, loss_fn, optimizer, loss_estimate, batch_no, t, epoch_no)
        test_loop(all_features_dataloader, model, loss_fn)
    print("Done!")
    save_model(model)

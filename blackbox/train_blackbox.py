import os
import sys
sys.path.append("/home/hjkim/RL_TimeSegment")

from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataloader import SeqComb
from models import Predictor
from torcheval.metrics.functional import binary_f1_score, multiclass_f1_score
from sklearn.metrics import f1_score

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--backbone', type=str)
parser.add_argument('--dataset', type=str)
args = parser.parse_args()
dataset = args.dataset
backbone = args.backbone


train_set = SeqComb.get_SeqComv(dataset, 'TRAIN')
valid_set = SeqComb.get_SeqComv(dataset, 'VALID')
test_set = SeqComb.get_SeqComv(dataset, 'TEST')

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d_in = 1 
seq_len = 100  
d_model = 64
d_out, average = SeqComb.get_num_classes(dataset)
lr = 1e-3
epochs = 120
save_path = "./best_{dataset}_{backbone}.pth"


model = Predictor.PredictorNetwork(d_in, d_model, d_out, seq_len, backbone)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

best_f1 = 0.0

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    train_preds, train_targets = [], []

    for batch in train_loader:
        x = batch['x']  # (B, T, C)
        y = batch['y']  # (B,)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        train_preds.append(preds.cpu())
        train_targets.append(y.cpu())

    # Train metrics
    avg_train_loss = running_loss / len(train_loader.dataset)
    train_preds = torch.cat(train_preds).numpy()
    train_targets = torch.cat(train_targets).numpy()
    train_acc = accuracy_score(train_targets, train_preds)
    train_f1 = f1_score(train_targets, train_preds, average=average)

    # Validation
    model.eval()
    valid_preds, valid_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_val = batch['x']
            y_val = batch['y']
            x_val, y_val = x_val.to(device), y_val.to(device)

            logits = model(x_val)
            preds = torch.argmax(logits, dim=1)
            valid_preds.append(preds.cpu())
            valid_targets.append(y_val.cpu())

    valid_preds = torch.cat(valid_preds).numpy()
    valid_targets = torch.cat(valid_targets).numpy()
    val_acc = accuracy_score(valid_targets, valid_preds)
    val_f1 = f1_score(valid_targets, valid_preds, average=average)

    print(
        f"Epoch {epoch:02d}/{epochs} "
        f"| Train Loss: {avg_train_loss:.4f} "
        f"| Train Acc: {train_acc:.4f} "
        f"| Train F1: {train_f1:.4f} "
        f"| Valid Acc: {val_acc:.4f} "
        f"| Valid F1: {val_f1:.4f}"
    )

    # Save best model
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_f1': best_f1,
        }, save_path.format(dataset=dataset, backbone=backbone))
        
        print(f"--> Saved new best LSTM model with Valid F1: {best_f1:.4f}")
        if best_f1 >= 1. - 1e-7:
            break

print(f"Training complete. Best Valid F1: {best_f1:.4f}")
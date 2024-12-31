import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scaler = GradScaler()

    def train(self, epoch, writer):
        self.model.train()
        running_loss = 0.0
        for batch, (inputs, labels) in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(inputs)
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(self.train_loader.dataset):>5d}], {(current/len(self.train_loader.dataset) * 100):>4f}%")
                step = epoch * len(self.train_loader.dataset) + current
                writer.add_scalar('training loss', loss, step)
        return running_loss / len(self.train_loader)
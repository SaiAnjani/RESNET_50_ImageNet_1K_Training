import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from data_loader import DataLoaderFactory
from model import ResNet50Model
from train import Trainer
from test import Tester

def main():
    train_dir = '/mnt/myvolume/Assign9/ILSVRC/Data/CLS-LOC/train'
    val_dir = '/mnt/myvolume/Assign9/ILSVRC/Data/CLS-LOC/val'
    batch_size = 128
    num_workers = 16
    num_epochs = 10
    start_epoch = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader_factory = DataLoaderFactory(train_dir, val_dir, batch_size, num_workers)
    train_loader, test_loader = data_loader_factory.get_data_loaders()

    resnet50_model = ResNet50Model(num_classes=1000)
    model = resnet50_model.get_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    writer = SummaryWriter(log_dir='logs')

    trainer = Trainer(model, train_loader, criterion, optimizer, device)
    tester = Tester(model, test_loader, criterion, device)

    for epoch in range(start_epoch, num_epochs):
        print(f'value of epoch is {epoch}')
        trainer.train(epoch, writer)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "params": {"batch_size": batch_size, "num_workers": num_workers}
        }
        os.makedirs(os.path.join("checkpoints", "resnet50"), exist_ok=True)
        torch.save(checkpoint, os.path.join("checkpoints", "resnet50", f"model_{epoch}.pth"))
        torch.save(checkpoint, os.path.join("checkpoints", "resnet50", f"checkpoint.pth"))
        lr_scheduler.step()
        tester.test(epoch + 1, writer, train_dataloader=train_loader, calc_acc5=True)

if __name__ == "__main__":
    main()
import torch
from tqdm import tqdm

class Tester:
    def __init__(self, model, test_loader, criterion, device):
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device

    def test(self, epoch, writer, train_dataloader=None, calc_acc5=False):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc="Testing", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        avg_loss = running_loss / len(self.test_loader)
        print(f'Test Error: \n Accuracy: {accuracy:.1f}%, Avg loss: {avg_loss:.4f} \n')
        writer.add_scalar('test accuracy', accuracy, epoch)
        writer.add_scalar('test loss', avg_loss, epoch)
        return avg_loss, accuracy
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import os

class Trainer:
    def __init__(self, epochs, lr, save_dir):
        self.epochs = epochs
        self.lr = lr
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=os.path.join(save_dir, "logs"))

    def fit(self, model, train_loader, val_loader):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            model.train()
            train_loss = 0
            
            for x, y in train_loader:
                optimizer.zero_grad()
                preds = model(x)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss = self.evaluate(model, val_loader, criterion)

            print(f"Epoch {epoch+1}/{self.epochs} → train={train_loss:.4f}, val={val_loss:.4f}")

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)

        model_path = os.path.join(self.save_dir, "model_final.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    @torch.no_grad()
    def evaluate(self, model, loader, criterion):
        model.eval()
        total_loss = 0
        for x, y in loader:
            preds = model(x)
            total_loss += criterion(preds, y).item()
        return total_loss / len(loader)

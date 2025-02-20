import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader.dataloadermodule import WildfireDataset
from models.Unet import UNet
from config import config
def train():
    dataset = WildfireDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = UNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):  
        for batch in dataloader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), config.paths["model_path"])
    print("Model training complete and saved.")

if __name__ == "__main__":
    train()

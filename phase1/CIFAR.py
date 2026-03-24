import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Dataset
train_data = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
test_data = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

# CNN Model (dimension verified)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3), nn.ReLU(),     
            nn.MaxPool2d(2, 2),                  
            nn.Conv2d(16, 32, 3), nn.ReLU(),     
            nn.MaxPool2d(2, 2),              

            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 128), nn.ReLU(),   
        )

    def forward(self, x):
        return self.model(x)

model = CNN().to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(1)
        total += y.size(0)
        correct += (preds == y).sum().item()

    train_acc = 100 * correct / total

    # Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            preds = outputs.argmax(1)

            total += y.size(0)
            correct += (preds == y).sum().item()

    test_acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
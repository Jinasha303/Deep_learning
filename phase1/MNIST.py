import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms 
from sklearn.metrics import confusion_matrix
# Load MNIST dataset
transform = transforms.ToTensor()
train_data= datasets.MNIST(root="./data", train = True, transform = transform, download = True)
test_data = datasets.MNIST(root="./data", train = False, transform = transform, download = True)
# Create data loaders
train_load= torch.utils.data.DataLoader(train_data, batch_size = 69, shuffle = True)
test_load =torch.utils.data.DataLoader(test_data, batch_size = 69, shuffle = False )
# Define the model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28,128), 
    nn.ReLU(),
    nn.Linear(128,10)
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(model.parameters(),lr = 0.001)
# Train the model
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms 
from sklearn.metrics import confusion_matrix
# Load MNIST dataset
transform = transforms.ToTensor()
train_data= datasets.MNIST(root="./data", train = True, transform = transform, download = True)
test_data = datasets.MNIST(root="./data", train = False, transform = transform, download = True)
# Create data loaders
train_load= torch.utils.data.DataLoader(train_data, batch_size = 69, shuffle = True)
test_load =torch.utils.data.DataLoader(test_data, batch_size = 69, shuffle = False )
# Define the model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28,128), 
    nn.ReLU(),
    nn.Linear(128,10)
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(model.parameters(),lr = 0.001)
# Train the model
accuracy_list = []
loss_list = []

epochs = 10

for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_load:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # accuracy calculation
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    accuracy_list.append(acc)
    loss_list.append(total_loss)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.2f}, Accuracy: {acc:.2f}%")
# Accuracy graph
plt.figure()
plt.plot(range(1, epochs+1), accuracy_list)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.show()

# Loss graph
plt.figure()
plt.plot(range(1, epochs+1), loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.show()
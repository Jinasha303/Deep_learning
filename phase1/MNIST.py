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
    nn.Linear(128,10),
    nn.ReLU()
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(model.parameters(),lr = 0.001)
# Train the model
epochs = 10
for epoch in range(epochs):
    total_loss = 0 

    for images, labels in train_load:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_load)}")
# Evaluate the model
correct = 0 
total = 0 
with torch.no_grad():
   for images , labels in test_load:
         outputs = model(images)
         _, predicted = torch.max(outputs.data, 1)
         total += labels.size(0)
         correct += (predicted == labels).sum().item()   
         print(f"TEst Accuracy: {100 * correct / total:.2f}%")
            

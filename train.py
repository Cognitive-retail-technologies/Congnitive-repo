# train.py with Early Stopping and Clear Messages

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# -----------------------
# SETTINGS
# -----------------------
TRAIN_DIR = "datasets/train"       # Folder with training images
VALID_DIR = "datasets/valid"       # Folder with validation images
MODEL_PATH = "model.pth"           # Where we will save the trained model
BATCH_SIZE = 16                     # How many images to process at once
EPOCHS = 50                         # Maximum number of times we go through all training images
LR = 0.001                          # Learning rate: how fast the model learns
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
 
PATIENCE = 5        
best_acc = 0    
counter = 0        

# -----------------------
# TRANSFORM IMAGES
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),             # Make all images same size
    transforms.ToTensor(),                     # Convert image to numbers
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])  # Adjust numbers to match pre-trained models
])

# -----------------------
# LOAD DATASET
# -----------------------
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=transform)
valid_ds = datasets.ImageFolder(VALID_DIR, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE)

print("Classes:", train_ds.classes)

# -----------------------
# LOAD MODEL
# -----------------------
model = models.resnet18(pretrained=True)       # Load pre-trained ResNet18
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_ds.classes))  # Change last layer for our classes
model = model.to(DEVICE)

# -----------------------
# LOSS AND OPTIMIZER
# -----------------------
criterion = nn.CrossEntropyLoss()             # Measures how wrong the model is
optimizer = optim.Adam(model.parameters(), lr=LR)  # Optimizer improves the model

# -----------------------
# TRAINING LOOP WITH EARLY STOPPING
# -----------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"\nEpoch {epoch+1}, Train Loss: {running_loss/len(train_loader):.4f}")

    # -----------------------
    # VALIDATION
    # -----------------------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")

    # -----------------------
    # EARLY STOPPING CHECK
    # -----------------------
    if acc > best_acc:
        best_acc = acc
        counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"🎉 New best model saved! Best Accuracy: {best_acc:.2f}%")
    else:
        counter += 1
        remaining = PATIENCE - counter
        print(f"⚠️ Validation did not improve. Counter: {counter}/{PATIENCE} "
              f"→ {remaining} epochs left before stopping.")

    if counter >= PATIENCE:
        print("\n⏹ Early stopping triggered. Training stopped.")
        break

print(f"\n✅ Training finished. Best Validation Accuracy: {best_acc:.2f}%")

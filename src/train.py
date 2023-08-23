import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import random


# Random seeds
def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, X_valid, X_test, y_train, y_valid = load_data(
    "src/fashion-mnist_train.csv", "src/fashion-mnist_test.csv"
)

data_transforms = get_transforms()

train_dataset = FashionMNIST(X=X_train, y=y_train, transform=data_transforms)
valid_dataset = FashionMNIST(X=X_valid, y=y_valid, transform=data_transforms)
test_dataset = FashionMNIST(X=X_test)

batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Cross entropy loss
loss = nn.CrossEntropyLoss()

# Adam optimiser
optimiser = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = lr_scheduler.CosineAnnealingLR(optimiser, T_max=N_EPOCHS)

loss_hist = []
val_loss_hist = []

# Loop over epochs
for epoch in range(N_EPOCHS):
    loss_acc = 0
    val_loss_acc = 0
    train_count = 0
    valid_count = 0

    # Loop over batches
    for imgs, labels in train_loader:
        # Reshape
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Forward pass
        preds = model(imgs)
        L = loss(preds, labels)

        # Backprop
        L.backward()

        # Update parameters
        optimiser.step()

        # Zero gradients
        optimiser.zero_grad()

        # Track loss
        loss_acc += L.detach().item()
        train_count += 1

    # Update learning rate
    scheduler.step()

    # Don't update weights
    with torch.no_grad():
        # Validate
        for val_imgs, val_labels in valid_loader:
            # Reshape
            val_imgs = val_imgs.to(device)
            val_labels = val_labels.to(device)

            # Forward pass
            val_preds = model(val_imgs)
            val_L = loss(val_preds, val_labels)

            # Track loss
            val_loss_acc += val_L.item()
            valid_count += 1

    # Save loss history
    loss_hist.append(loss_acc / train_count)
    val_loss_hist.append(val_loss_acc / valid_count)

    # Print loss
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1}/{N_EPOCHS}, loss {loss_acc/train_count:.5f}, val_loss {val_loss_acc/valid_count:.5f}"
        )

print("")
print("Training complete!")

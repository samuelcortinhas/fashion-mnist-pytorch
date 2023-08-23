import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from dataset import FashionMNIST, load_data
from model import ConvNet


def train(
    args,
    model,
    device,
    train_loader,
    valid_loader,
    criterion,
    optimiser,
    scheduler,
    n_epochs=10,
    verbose=True,
):
    # Set to training mode
    model.train()

    # Loop over epochs
    for epoch in range(n_epochs):
        # Loop over batches
        for imgs, labels in train_loader:
            # Reshape + send to device
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Forward pass
            preds = model(imgs)
            L = criterion(preds, labels)

            # Backprop
            L.backward()

            # Update parameters
            optimiser.step()

            # Zero gradients
            optimiser.zero_grad()

            # Track loss
            loss = L.detach().item()

        # Update learning rate
        scheduler.step()

        # Don't update weights
        with torch.no_grad():
            # Set to evaluation mode
            model.eval()

            # Validate
            for val_imgs, val_labels in valid_loader:
                # Reshape
                val_imgs = val_imgs.to(device)
                val_labels = val_labels.to(device)

                # Forward pass
                val_preds = model(val_imgs)
                val_L = criterion(val_preds, val_labels)

                # Track loss
                val_loss = val_L.item()

        # Print loss
        if verbose:
            print(
                f"Epoch {epoch+1}/{n_epochs}, loss {loss:.5f}, val_loss {val_loss:.5f}"
            )


# Random seeds
def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def main():
    seed = 0
    batch_size = 64
    learning_rate = 0.001
    n_epochs = 10
    conv1_channels = 64
    conv2_channels = 128
    linear1_size = 256
    linear2_size = 128
    dropout_rate = 0.25
    verbose = True
    train_path = "src/fashion-mnist_train.csv"
    test_path = "src/fashion-mnist_test.csv"
    debug = True

    # GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    set_seed(seed=seed)

    # Data
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(
        train_path=train_path, test_path=test_path, debug=debug
    )

    # Data augmentations (Having problems with torchvision on my pc -> skipping for now)
    # data_transforms = get_transforms()

    # Define datasets
    train_dataset = FashionMNIST(X=X_train, y=y_train)
    valid_dataset = FashionMNIST(X=X_valid, y=y_valid)
    test_dataset = FashionMNIST(X=X_test)

    # Define dataloaders
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        dataset=valid_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = ConvNet(
        conv1_channels=conv1_channels,
        conv2_channels=conv2_channels,
        linear1_size=linear1_size,
        linear2_size=linear2_size,
        dropout_rate=dropout_rate,
    )

    # Define loss, optimiser and scheduler
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_epochs)

    # Train model
    train(
        args=None,
        model=model,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimiser=optimiser,
        scheduler=scheduler,
        n_epochs=n_epochs,
        verbose=verbose,
    )

    # Save model
    torch.save(
        {
            "epoch": n_epochs,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        "trained_convnet_v1.pt",
    )


if __name__ == "__main__":
    main()

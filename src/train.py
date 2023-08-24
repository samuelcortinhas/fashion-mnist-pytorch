import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from dataset import FashionMNIST, load_data
from model import ConvNet

# Load environment variables
load_dotenv()


def train(
    cfg,
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

            # Logging
            if cfg["logging"]:
                wandb.log({"loss": loss})

        # Logging
        if cfg["logging"]:
            wandb.log({"learning_rate": scheduler.get_lr(), "epoch": epoch})

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
                if cfg["logging"]:
                    wandb.log({"val_loss": val_loss})

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
    # Configuration
    with open("config.json", "r") as f:
        cfg = json.load(f)

    if cfg["logging"]:
        # Sign-in
        api_key = os.environ["WANDB_API_KEY"]
        wandb_entity = os.environ["WANDB_ENTITY"]
        wandb.login(key=api_key)

    # GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    set_seed(seed=cfg["seed"])

    # Data
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(
        train_path=cfg["train_path"], test_path=cfg["test_path"], debug=cfg["debug"]
    )

    # Data augmentations (Having problems with torchvision on my pc -> skipping for now)
    # data_transforms = get_transforms()

    # Define datasets
    train_dataset = FashionMNIST(X=X_train, y=y_train)
    valid_dataset = FashionMNIST(X=X_valid, y=y_valid)
    test_dataset = FashionMNIST(X=X_test)

    # Define dataloaders
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=cfg["batch_size"], shuffle=True
    )
    valid_loader = DataLoader(
        dataset=valid_dataset, batch_size=cfg["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=cfg["batch_size"], shuffle=False
    )

    # Model
    model = ConvNet(
        conv1_channels=cfg["conv1_channels"],
        conv2_channels=cfg["conv2_channels"],
        linear1_size=cfg["linear1_size"],
        linear2_size=cfg["linear2_size"],
        dropout_rate=cfg["dropout_rate"],
    )

    # Define loss, optimiser and scheduler
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(params=model.parameters(), lr=cfg["learning_rate"])
    scheduler = lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg["n_epochs"])

    # Start run
    if cfg["logging"]:
        run = wandb.init(
            entity=wandb_entity,
            project="fashion-mnist-pytorch",
            config=cfg,
            save_code=True,
        )

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
        n_epochs=cfg["n_epochs"],
        verbose=cfg["verbose"],
    )

    # End run
    if cfg["logging"]:
        run.finish()

    # Save model
    torch.save(
        {
            "epoch": cfg["n_epochs"],
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        "models/trained_convnet_v1.pt",
    )


if __name__ == "__main__":
    main()

# To do next:
# Add logging with wandb
# Parse CLI arguments or config file or both (maybe cfg for model architecture and cli for training args)
# Add unit tests using torch.testing
# Add evaluation.ipynb to make some plots and calc. metrics
# Add route to fastapi app for model inference - test locally
# Add docs and examples to readme

import torch

from model import ConvNet


def load_model(model_path):
    # GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialise model and optimiser
    model = ConvNet().to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path)

    # Load states
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluation mode
    model.eval()

    return model


def inference(model, imgs):
    # GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        imgs = imgs.to(device)
        output = model(imgs)

        _, preds = torch.max(output, 1)

        return preds

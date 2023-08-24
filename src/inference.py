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


def inference(model, img):
    # Classes
    classes = [
        "0 T-shirt/top",
        "1 Trouser",
        "2 Pullover",
        "3 Dress",
        "4 Coat",
        "5 Sandal",
        "6 Shirt",
        "7 Sneaker",
        "8 Bag",
        "9 Ankle boot",
    ]

    # GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        img = img.to(device)
        output = model(img)

        _, preds = torch.max(output, 1)

        return [classes[i] for i in preds]

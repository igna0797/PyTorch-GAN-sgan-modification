import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from sgan import Discriminator
from utils import parseArguments


def load_dataset(args) -> DataLoader:
    """Load the MNIST dataset with the appropriate transformations."""
    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.MNIST(
        root="../../data/mnist",
        train=True,
        download=True,
        transform=transform
    )

    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


def load_model(weights_path: str, device: torch.device) :
    """Load the Discriminator model and its weights."""
    discriminator = Discriminator()
    discriminator.to(device)

    try:
        checkpoint = torch.load(weights_path, map_location=device)
        discriminator.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        exit(1)

    discriminator.eval()  # Set to evaluation mode
    return discriminator


def evaluate_discriminator(discriminator: Discriminator, dataloader: DataLoader, device: torch.device) :
    """Evaluate the Discriminator on the dataset."""
    correct_predictions = 0
    total_samples = 0
    falseNegatives  = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            validity, label_outputs = discriminator(images)
            label_probabilities = label_outputs[:, :-1]  # Exclude the last value (label for 'real/fake')
            predicted_labels = torch.argmax(label_probabilities, dim=1)

            correct_predictions += (predicted_labels == labels).sum().item()
            falseNegatives += torch.sum(tensor == 0).item()
            total_samples += labels.size(0)

    accuracy = 100 * correct_predictions / total_samples
    falseNegativesPerrcentage = (falseNegatives / total_samples) * 100

    return accuracy , falseNegativesPerrcentage


if __name__ == "__main__":
    opt = parseArguments()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataloader = load_dataset(opt)

    # Load model
    discriminator = load_model(opt.weights_path, device)

    # Evaluate model
    accuracy, falseNegativesPerrcentage = evaluate_discriminator(discriminator, dataloader, device)

    print(f"Discriminator accuracy on the MNIST dataset: {accuracy:.2f}%")
    print(f"False Negatives Percentage: {falseNegativesPerrcentage:.2f}%")

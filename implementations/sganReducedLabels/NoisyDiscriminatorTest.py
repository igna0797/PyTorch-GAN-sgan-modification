import os
import torch
import torchvision.transforms as transforms
import pickle
from torch.utils.data import DataLoader
from torchvision import datasets
from sgan import Discriminator
from utils import parseArguments , NoiseAdder , get_opt_path
from torchvision.utils import save_image


def load_dataset( args ) -> DataLoader:
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

    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False,drop_last=True)


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


def evaluate_discriminator(discriminator: Discriminator, dataloader: DataLoader, device: torch.device, opt) :
    """Evaluate the Discriminator on the dataset."""
    correct_predictions = 0
    total_samples = 0
    i=0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device).float()  
            labels = labels.to(device)
            noisy_images, noise_labels = NoiseAdder.add_noise(images,opt)

            noise_labels = noise_labels.to(device) 
            noisy_images = noisy_images.to(device).float()  # Convert to float before feeding to the model    

            _, label_outputs = discriminator(noisy_images)
            label_probabilities = label_outputs[:, :-1]  # Exclude the last value (label for 'real/fake')
            predicted_labels = torch.argmax(label_probabilities, dim=1)

            correct_predictions += torch.logical_or(predicted_labels == labels, predicted_labels == noise_labels).sum().item()
            total_samples += labels.size(0)
            if i == 0:
                print(f'samples: {labels.size(0)}, labels size: {labels.size()}')
            #Imprime la imagen con el ruido y sus labels
            i += 1
            if i%100 == 0 or i < 10:
                save_path = f'Imagen_numero_{i}.png' # Define the path to save the image
                log_file_path = "output_log.txt"   # Define the path to the log file
                save_image(images[0], 'original'+save_path, normalize=True)
                save_image(noisy_images[0], save_path, normalize=True)
            
                log_message = f'label de la imagen {i}: {labels[0]}, label de el ruido {i}: {noise_labels[0]}, label predecido {i}: {predicted_labels[0]}'
                with open(log_file_path, 'a') as f:
                    f.write(log_message + '\n')

    accuracy = 100 * correct_predictions / total_samples
   
    return accuracy


if __name__ == "__main__":
    CallerOptions = parseArguments()
    optionsPath = get_opt_path(__file__, weights_path= CallerOptions.weights_path)
    try:
        with open(optionsPath, "rb") as f:
            opt = pickle.load(f)
    except (FileNotFoundError, IOError, pickle.UnpicklingError) as e:
        # Handle the exception by printing an error message or providing default values
        print(f"An error occurred: {e}")
        print(f"optionsPath: {optionsPath}")
        raise
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataloader = load_dataset(opt)

    # Load model
    discriminator = load_model(CallerOptions.weights_path, device)

    # Evaluate model
    accuracy = evaluate_discriminator(discriminator, dataloader, device, opt)

    print(f"Discriminator accuracy on the MNIST dataset: {accuracy:.2f}%")

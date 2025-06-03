import os
import torch
import torchvision.transforms as transforms
import pickle
from torch.utils.data import DataLoader
from torchvision import datasets
from sgan import Discriminator
from utils import parseArguments, NoiseAdder, get_opt_path, labelEncoder
from torchvision.utils import save_image
from NoisyDiscriminatorTest import load_dataset , load_model

def evaluate_discriminator_correct_guesses(discriminator: Discriminator, dataloader: DataLoader, device: torch.device, opt):
    """Evaluate the Discriminator and count correct guesses excluding FAKE labels."""
    correct_predictions = 0
    total_samples = 0
    
    # Initialize encoder
    Encoder = labelEncoder(num_classes=10)
    
    # Get indices of encoded labels that don't contain FAKE (10)
    labels_map = Encoder.get_indexMap()
    non_fake_indices = [encoded_idx for (num1, num2), encoded_idx in labels_map.items() 
                       if 10 not in (num1, num2)]

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device).float()  
            labels = labels.to(device)
            
            # Get discriminator predictions
            validity, label_outputs = discriminator(images)
            
            # Filter predictions to exclude FAKE labels
            non_fake_predictions = label_outputs[:, non_fake_indices]
            
            # Get maximum predictions among non-fake labels
            max_values, max_indices_subset = torch.max(non_fake_predictions, dim=1)
            
            # Convert back to original encoded label indices
            predicted_encoded_labels = torch.tensor([non_fake_indices[idx] for idx in max_indices_subset]).to(device)
            
            # Decode predictions to get original number pairs
            predicted_pairs = [Encoder.reverse_map[idx.item()] for idx in predicted_encoded_labels]
            
            # Check if real labels match any number in the predicted pairs
            for i, real_label in enumerate(labels):
                predicted_pair = predicted_pairs[i]
                if real_label.item() in predicted_pair:
                    correct_predictions += 1
                total_samples += 1

    accuracy = 100 * correct_predictions / total_samples
    return correct_predictions, total_samples, accuracy


if __name__ == "__main__":
    CallerOptions = parseArguments()
    optionsPath = get_opt_path(__file__, weights_path=CallerOptions.weights_path)
    
    try:
        with open(optionsPath, "rb") as f:
            opt = pickle.load(f)
    except (FileNotFoundError, IOError, pickle.UnpicklingError) as e:
        print(f"An error occurred: {e}")
        print(f"optionsPath: {optionsPath}")
        raise
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset - using the existing function from NoisyDiscriminatorTest
    dataloader = load_dataset(opt)

    # Load model - using the existing function from NoisyDiscriminatorTest
    discriminator = load_model(CallerOptions.weights_path, device)

    # Evaluate model for correct guesses
    correct_guesses, total_samples, accuracy = evaluate_discriminator_correct_guesses(discriminator, dataloader, device, opt)

    print(f"Total samples: {total_samples}")
    print(f"Correct guesses (excluding FAKE labels): {correct_guesses}")
    print(f"Accuracy: {accuracy:.2f}%")
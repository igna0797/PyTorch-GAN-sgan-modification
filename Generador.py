import torch
import os
from torchvision.utils import save_image
from sgan import Generator  # Import your Generator class from Sgan.py

def generate_image_from_seed(seed, save_path=None):
    # Set a fixed seed for the random number generator (for reproducibility)
    torch.manual_seed(seed)  # You can use any seed value you like

    # Initialize the generator model
    generator = Generator()

    # Load the pre-trained generator weights
    generator_weights_path = "/content/drive/MyDrive/Redes neuronales/Monografia/generator_weights.pth"

    generator.load_state_dict(torch.load(generator_weights_path))

    # Set the generator in evaluation mode
    generator.eval()

    # Generate an image from the seed
    with torch.no_grad():
        latent_dim = 100  # Adjust the latent dimension as needed
        noise = torch.FloatTensor(1, latent_dim).normal_(0, 1)  # Generate random noise
        #noise[:, :len(seed)] = torch.FloatTensor(seed)  # Replace the beginning of the noise with the seed
        generated_image = generator(noise)

    # If save_path is provided, save the generated image
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_image(generated_image, save_path, normalize=True)

    # Return the generated image
    return generated_image

# Example usage:
# Example usage with different seeds:

seed_value = 1  # Replace with your desired seed value
output_dir = "/content/drive/MyDrive/Redes neuronales/Monografia/images/"  # Replace with your desired output directory
output_filename = f"seed_{seed_value:.3f}.png"  # Access the first element of the list
output_path = os.path.join(output_dir, output_filename)

# Generate images with different seeds
generated_image = generate_image_from_seed(seed_value,output_path)

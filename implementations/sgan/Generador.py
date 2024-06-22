import torch
import os
import argparse

from torchvision.utils import save_image
from sgan import Generator  # Import your Generator class from Sgan.py

def generate_image_from_seed(seed, save_path=None,generator_weights_path="../../trainings/n-lineas_0_Random_False/generator_weights.pth"):
    # Set a fixed seed for the random number generator (for reproducibility)
    torch.manual_seed(seed)  # You can use any seed value you like

    # Initialize the generator model
    generator = Generator()

    cuda = True if torch.cuda.is_available() else False
    # Load the pre-trained generator weights
    if cuda == True :
        generator.load_state_dict(torch.load(generator_weights_path))
    else :
        map_location=torch.device('cpu') 
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o ", "--output_path", type=str, default="../../images/", help="directory for the Image returned by the generator")
    parser.add_argument("-s","--seed" ,type=int , default=3 ,help="seed for the generator for reproducible results")
    parser.add_argument("-w ", "--weights_path", type=str, default="../../trainings/n-lineas_3_Random_False/generator_weights.pth", help="directory for the weigths of the generator")
    arguments = parser.parse_args()
    #old directory : "/content/drive/MyDrive/Redes neuronales/Monografia/images/"
    abs_dir = os.path.dirname(os.path.abspath(__file__))
    
    seed_value = arguments.seed
    output_dir = os.path.join(abs_dir, arguments.output_path)
    output_filename = f"seed_{seed_value:.3f}.png"  # Access the first element of the list
    output_path = os.path.join(output_dir, output_filename)
    print(output_path)
    # Generate images with different seeds
    generator_weights_path = os.path.join(abs_dir, arguments.weights_path)
    generated_image = generate_image_from_seed(seed_value,output_path,generator_weights_path)

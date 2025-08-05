import torch
import os
from torchvision.utils import save_image
import argparse
import pickle


from sgan import Generator
from utils import parseArguments, NoiseAdder, get_opt_path, labelEncoder
from NoisyDiscriminatorTest import load_dataset 

def load_generator(weights_path: str, device: torch.device):
    generator = Generator().to(device)
    generator.eval()
    generator.load_state_dict(torch.load(weights_path, map_location=device))
    return generator

def generate_images_to_folder(generator, latent_dim, batch_size, num_images, save_folder, device):
    os.makedirs(save_folder, exist_ok=True)
    n_batches = num_images // batch_size

    with torch.no_grad():
        for i in range(n_batches):
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_imgs = generator(z)
            for j in range(batch_size):
                save_path = os.path.join(save_folder, f"img_{i * batch_size + j:05d}.png")
                save_image(gen_imgs[j], save_path, normalize=True)


def save_real_images_to_folder(dataloader, save_path: str, num_images: int = 10000):
    os.makedirs(save_path, exist_ok=True)
    count = 0

    for batch_imgs, _ in dataloader:
        for img in batch_imgs:
            if count >= num_images:
                return
            save_image(img, os.path.join(save_path, f"img_{count:05d}.png"), normalize=True)
            count += 1

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

    abs_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(abs_dir, CallerOptions.output_path)
    gen_output_dir = os.path.join(output_dir, "GeneratorImages")
    Real_output_dir = os.path.join(output_dir, "RealImages")

    print(gen_output_dir)
    print(Real_output_dir)
    
    generator_weights_path = os.path.join(abs_dir, CallerOptions.weights_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = load_generator(generator_weights_path, device)

    # Generate fake images
    generate_images_to_folder(generator, latent_dim=100, batch_size=64, num_images=10000,
                            save_folder=gen_output_dir, device=device)

    save_real_images_to_folder(dataloader, Real_output_dir, num_images = 10000)
  


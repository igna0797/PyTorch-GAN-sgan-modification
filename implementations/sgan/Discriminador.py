import os
import torch
import torch.nn as nn
import argparse

from torchvision.transforms import transforms
from torchvision.utils import save_image
from sgan import Discriminator  # Import your Discriminator class from Sgan.py
from sgan import opt
from PIL import Image


def load_discriminator_weights(discriminator, weights_path):
    # Load the saved discriminator weights
    try:
        discriminator.load_state_dict(torch.load(weights_path))
        print("Discriminator weights loaded successfully.")
    except:
        print("Error loading discriminator weights. Make sure the path is correct.")

def evaluate_image_with_discriminator(discriminator, image):
    # Set the discriminator to evaluation mode
    discriminator.eval()

    # Preprocess the input image (you can use the same transformations as in your training data)
    transform = transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.ToTensor(),
        # Add any other necessary transformations here
    ])
    image = transform(image).unsqueeze(0)  # Add a batch dimension

    # Forward pass through the discriminator
    with torch.no_grad():
        validity, label = discriminator(image)

    # You can return the discriminator's output (validity) as needed
    return label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w ", "--weights_path", type=str, default=".", help="directory for the weigths of the discriminator")
    parser.add_argument("-i","--image_path" ,type=str , default="images/seed_1.000.png",help="directory for the image to discriminate")
    arguments = parser.parse_args()
    #old directory for weights:/content/drive/MyDrive/Redes neuronales/Monografia/discriminator_weights.pth
    print(arguments)
    weights_path = arguments.weights_path
    # Create an instance of the Discriminator class
    discriminator = Discriminator()

    # Load the discriminator weights
    load_discriminator_weights(discriminator, weights_path)
    
    # Load Image and keeps it grayscale
    image_path = arguments.image_path
    image = Image.open(image_path).convert("L") 

    label_probabilities = evaluate_image_with_discriminator(discriminator, image)
   
    # Exclude the last choice (opt.num_classes)
    label_probabilities = label_probabilities[:, :-1]

    # Find the label with the highest probability
    label = torch.argmax(label_probabilities, dim=1).item()
    print(f"El numero es un \" {label} \" ")
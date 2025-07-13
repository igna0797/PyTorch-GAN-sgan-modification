import os
import torch
import torchvision.transforms as transforms
import pickle
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets
from torchvision.utils import save_image
from collections import defaultdict
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sgan import Discriminator, Generator
from utils import parseArguments , NoiseAdder , get_opt_path , labelEncoder
from Generador import generate_image_from_seed

def load_dataset( args ) -> DataLoader:
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
    discriminator = Discriminator()
    if cuda:
      discriminator.cuda()
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        discriminator.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        exit(1)

    discriminator.eval()
    return discriminator

def load_generator(generator_weights_path:str,device):
    if not os.path.exists(generator_weights_path):
        raise FileNotFoundError(f"Generator weights file not found at: {generator_weights_path}")
    # Initialize the generator model
    generator = Generator()
    if cuda:
      generator.cuda()
    # Load the pre-trained generator weights
    generator.load_state_dict(torch.load(generator_weights_path,map_location=device))
    # Set the generator in evaluation mode
    generator.eval()
    return generator

def generate_images(generator,opt,device):
        batch_size = opt.batch_size
        latent_dim = opt.latent_dim 
        fake_label_list = [opt.num_classes] * batch_size
        #FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        #generate images
        with torch.no_grad():
            latent_dim = 100  # Adjust the latent dimension as needed
            z = Variable(torch.tensor(np.random.normal(0, 1, (batch_size, opt.latent_dim)),dtype=torch.float32, device=device))
            generated_images = generator(z)
        return generated_images , fake_label_list

def evaluate_discriminator(discriminator: Discriminator, generator: Generator ,dataloader: DataLoader, device: torch.device, opt) :
    total_samples = 0
    falseNegatives = 0
    falsePositive = 0
    i = 0

    label_pred_counts = defaultdict(int)
    label_pred_correct = defaultdict(int)
    encoder = labelEncoder(num_classes=10)
    all_label_indices = sorted(encoder.index_map.values())
    
    #Create a dictionary with all the real values in which I add in each how many times each prediction was chosen.
    confusion = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        #------Real Images ------- 
        for images, labels in dataloader:
            #Load images and add noise
            images = images.to(device).float()
            labels = labels.to(device)
            noisy_images, noise_labels = NoiseAdder.add_noise(images, opt)
            noise_labels = noise_labels.to(device)
            noisy_images = noisy_images.to(device).float()
            
            #Run discriminator chose the highest choise and decode it
            validity , label_outputs = discriminator(noisy_images)
            predicted_combination = torch.argmax(label_outputs, dim=1)
            pred_true_labels, pred_noise_labels = encoder.decode_labels(predicted_combination)


            falseNegatives += torch.sum(validity == 0).item()
            total_samples += labels.size(0)
            for j in range(labels.size(0)):
                true_combo = tuple(sorted((labels[j].item(), noise_labels[j].item())))
                pred_combo = tuple(sorted((pred_true_labels[j], pred_noise_labels[j])))
                confusion[true_combo][pred_combo] += 1
            

            #Logs
            if i == 0:
                print(f'samples: {labels.size(0)}, labels size: {labels.size()}')
            i += 1
            if i % 100 == 0 or i < 10:
                save_path = f'Imagen_numero_{i}.png'
                log_file_path = "output_log.txt"
                save_image(images[0], 'original' + save_path, normalize=True)
                save_image(noisy_images[0], save_path, normalize=True)
                log_message = f'label de la imagen {i}: {labels[0]}, label de el ruido {i}: {noise_labels[0]}, combinacion predicha {i}: ({pred_true_labels[0]},{pred_noise_labels[0]})'
                with open(log_file_path, 'a') as f:
                    f.write(log_message + '\n')
                    
            #------ Generated iamges -------
            gen_noisy_imgs , fake_aux_labels = generate_images(generator,opt,device)
            #adds noise
            gen_noisy_imgs,  noise_label = NoiseAdder.add_noise(gen_noisy_imgs, opt)          
            # This also commpresses (FAKE,2) and (2,FAKE) into one single label
            #fake_aux_gt = encoder.encode_labels(fake_aux_labels, noise_label)  # Encode fake labels with noise labels
      
            fake_validity , fake_label_output = discriminator(gen_noisy_imgs)
            predicted_fake_combination = torch.argmax(fake_label_output, dim=1)
            pred_fake_labels, pred_fake_noise_labels = encoder.decode_labels(predicted_fake_combination)
                
            falsePositive += torch.sum(fake_validity == 1).item()
            total_samples += fake_label_output.size(0)
            for j in range(len(fake_aux_labels)):
                true_combo = tuple(sorted((fake_aux_labels[j], noise_label[j].item())))
                pred_combo = tuple(sorted((pred_fake_labels[j], pred_fake_noise_labels[j])))
                confusion[true_combo][pred_combo] += 1

        #Logs?
                    
    falseNegativesPerrcentage = (falseNegatives / total_samples) * 100

    print(f"False Negatives Percentage: {falseNegativesPerrcentage:.2f}%\n")

# Plot counts
    plot_confusion_from_dict(
        confusion,
        title="Confusion Matrix - Counts",
        filename="confusion_matrix.png",
        percentage=False
    )

    # Plot percentages
    plot_confusion_from_dict(
        confusion,
        title="Confusion Matrix (%) - Prediction Distribution",
        filename="confusion_matrix_percentage.png",
        percentage=True
    )
    return  falseNegativesPerrcentage

def plot_confusion_from_dict(confusion_dict, 
                              title="Confusion Matrix", 
                              filename="confusion_matrix.png", 
                              percentage=False):
    """
    Build and plot the confusion matrix from a nested confusion dictionary.

    Args:
        confusion_dict (dict): Format {true_label: {pred_label: count}}
        title (str): Plot title.
        filename (str): Output image filename.
        percentage (bool): Normalize rows to percentages if True.
    """
    # Extract labels
    true_combos = sorted(confusion_dict.keys())
    pred_combos = sorted({k for v in confusion_dict.values() for k in v.keys()})

    # Initialize matrix
    matrix = np.zeros((len(true_combos), len(pred_combos)), dtype=np.float32)

    # Fill matrix
    for i, true_label in enumerate(true_combos):
        for j, pred_label in enumerate(pred_combos):
            matrix[i, j] = confusion_dict[true_label].get(pred_label, 0)

    # Normalize if percentage
    if percentage:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, where=row_sums != 0) * 100
        fmt = ".1f"
        cbar_label = "Percentage (%)"
        cmap = "viridis"
    else:
        fmt = "g"
        cbar_label = "Count"
        cmap='Blues'

    # Plot
    plt.figure(figsize=(16, 16))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        xticklabels=pred_combos,
        yticklabels=true_combos,
        cmap= cmap,
        cbar_kws={'label': cbar_label}
    )

    plt.xlabel("Predicted Label (pred1, pred2)")
    plt.ylabel("True Label (true1, noise2)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Image saved in {filename}")

    return matrix, true_combos, pred_combos

if __name__ == "__main__":
    CallerOptions = parseArguments()
    optionsPath = get_opt_path(__file__, weights_path= CallerOptions.weights_path)
    weight_dir = os.path.dirname(CallerOptions.weights_path)
    generator_weights_path = os.path.join(weight_dir, 'generator_weights.pth')
    discriminator_weights_path = os.path.join(weight_dir, 'discriminator_weights.pth')
    print(f"Disc{discriminator_weights_path}")
    print(f"Gen{generator_weights_path}")

    try:
        with open(optionsPath, "rb") as f:
            opt = pickle.load(f)
    except (FileNotFoundError, IOError, pickle.UnpicklingError) as e:
        print(f"An error occurred: {e}")
        print(f"optionsPath: {optionsPath}")
        raise
    global cuda
    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if cuda==True else "cpu")
    dataloader = load_dataset(opt)
    discriminator = load_model(discriminator_weights_path, device)
    generator = load_generator(generator_weights_path, device)
    evaluate_discriminator(discriminator,generator, dataloader, device, opt)

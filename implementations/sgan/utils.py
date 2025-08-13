import argparse
import os
import numpy as np
import torch
import itertools as it

from torchvision import transforms, datasets
from torchvision.utils import save_image 

mnist_loader = None

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    parser.add_argument("--noise_type", type=str, choices=["lines", "gaussian", "mnist"], help="Type of noise to add: 'lines', 'gaussian', 'mnist'")
    parser.add_argument("--max_lines", type=int, default=3, help="number of lines added as noise")
    parser.add_argument("--random_amount_lines", type=bool, default= False , help="if false always maximum amount")
    parser.add_argument("--image_output" ,type=str ,help="Directory to store the images generated during training")
    parser.add_argument("--Training_output" ,type=str ,help="Directory to store the training")
    parser.add_argument("--partialMatchFlag", type=bool, default=True, help="To use the partial match when comparing discriminator and generator") 
# This is for loading a already done model
    parser.add_argument("-w ", "--weights_path", type=str, help="directory for the weigths of the generator")
    parser.add_argument("-o ", "--output_path", type=str, default="images/", help="directory for the Image returned by the generator")
    parser.add_argument("-s","--seed" ,type=int , default=3 ,help="seed for the generator for reproducible results")
    parser.add_argument("-i","--image_path" ,type=str , default="implementations/sgan/images/seed_1.000.png",help="directory for the image to discriminate")
    opt = parser.parse_known_args()[0]
    print(opt)
    return opt

def get_directory(__file__,output):
  script_dir = os.path.dirname(os.path.abspath(__file__))
  output_directory = os.path.basename(os.path.normpath(output))
  directory = os.path.join(script_dir, "../../trainings/" + output_directory)   
  return directory
"""
def get_directory(__file__,max_lines=3 , random_amount_lines = False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the relative path to the pickle file
    directory = os.path.join(script_dir, "../../trainings/n-lineas_" + str(max_lines) + "_Random_"+ str(random_amount_lines))    
    #directory = "../../../content/drive/MyDrive/Redes neuronales/Monografia/n-lineas_" + str(opt.max_lines) + "_Random_"+ str(opt.random_amount_lines)
    return directory
    """
def get_opt_path(__file__ , weights_path):
    abs_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path to the pickle file
    opt_path = os.path.join(abs_dir, os.path.dirname(weights_path), 'opt.pkl')
    #directory = "../../../content/drive/MyDrive/Redes neuronales/Monografia/n-lineas_" + str(opt.max_lines) + "_Random_"+ str(opt.random_amount_lines)
    return opt_path


class NoiseAdder:
    """Class that adds noise and stores the mnist_loader to reuse it across calls."""
    
    mnist_loader = None  # Class-level attribute to cache the loader
    
    @staticmethod
    def add_noise(images, args):
        """Add noise to the images based on the specified noise type."""
        
        if args.noise_type == "lines":  # Lines noise
            return add_lines(images, max_amount_lines=args.max_lines, random_amount_lines=args.random_amount_lines)
        
        elif args.noise_type == "mnist":  # MNIST noise
            # Initialize the mnist_loader only if it is not already created
            if NoiseAdder.mnist_loader is None:
                NoiseAdder.mnist_loader = get_mnist_loader(images, args)
            
            # Add MNIST noise to the images
            return add_mnist_noise(images, NoiseAdder.mnist_loader)
        
        else:
            raise ValueError(f"Unknown noise type: {args.noise_type}")


def get_mnist_loader(images, args):
    os.makedirs("../../data/mnist2", exist_ok=True)
    transform = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5] )
        ])
    mnist_data = datasets.MNIST(root="../../data/mnist2", train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(mnist_data, batch_size=images.size(0), shuffle=True,drop_last=True)
    
def add_mnist_noise(images, mnist_loader):
    if len(images.shape) == 3:# Single image case
        next_data = next(iter(mnist_loader))
        #sample_images, _ = next_data
        #save_image(sample_images[:20],  "dataset_visualizationMNISNST.png", nrow=5, normalize=True)
        noise_images, noise_labels  = next_data
        noise_images = noise_images.to(images.device).float()  # Convert to float and match device
        #print(f"rudio {noise_image.shape}")
        #print(f"imagenes {images.shape}")
        noise_images = noise_images[0]
        noise_images = noise_images.expand_as(images)  # Expand to match input image channels
        noise_images = torch.maximum(noise_images , images) 
    elif len(images.shape) == 4:  # Batch image case
        next_data = next(iter(mnist_loader))
        noise_images, noise_labels  = next_data
        noise_images = noise_images.to(images.device).float()  # Convert to float and match device
        noise_images = noise_images.expand_as(images)  # Expand to match input image channels
        noise_images = torch.maximum(noise_images , images)

    return noise_images , noise_labels 

def add_lines(images,max_amount_lines=1, random_amount_lines=False):
    if random_amount_lines == True:
      number_of_lines = np.random.randint(1,max_amount_lines+1)
    else:
      number_of_lines = max_amount_lines
#@decicion: en cada batch la cantidad de linea sagregadas es igual puede variar de batch en batch pero en uno solo se mantiene
    if len(images.shape) == 3:  # Single image case
        channels, height, width = images.shape
        images_with_lines = images.clone()  # Create a copy to work with
        for _ in range(number_of_lines):
          # Add horizontal line
          horizontal_line_pos = np.random.randint(0, height)
          images_with_lines[:, horizontal_line_pos, :] = 1  # Change pixel values to black

          # Add vertical line
          vertical_line_pos = np.random.randint(0, width)
          images_with_lines[:, :, vertical_line_pos] = 1  # Change pixel values to black
   
    elif len(images.shape) == 4:  # Batch image case
      batch_size, channels, height, width = images.shape
      images_with_lines = images.clone()  # Create a copy to work with
      for i in range(batch_size):
          for _ in range(number_of_lines):
            # Add horizontal line
            horizontal_line_pos = np.random.randint(0, height)
            images_with_lines[i, :, horizontal_line_pos, :] = 1  # Change pixel values to black

            # Add vertical line
            vertical_line_pos = np.random.randint(0, width)
            images_with_lines[i, :, :, vertical_line_pos] = 1  # Change pixel values to black

    return images_with_lines


class labelEncoder:
    def __init__(self,num_classes):
        """Initialize the TupleIndexer with the second_iterable."""
        self.num_classes = num_classes 
        possible_labels = list(range(num_classes+1))
        self.new_label_space = it.combinations_with_replacement(possible_labels, 2)
        self.index_map = {}
        self.reverse_map = {}
        for idx, tup in enumerate(self.new_label_space):
            self.index_map[tup] = idx
            self.reverse_map[idx] = tup
        self.index_map.pop((num_classes,num_classes)) #Removing (FAKE, FAKE) as it is not a valid combination
        self.reverse_map.pop(len(self.index_map))  # Removing (FAKE, FAKE) as it is not a valid combination

    def encode_labels(self, label1 , label2 ) -> list :
        """Returns indices of tuples in first_list based on the index_map."""
        if isinstance(label1, torch.Tensor):
            label1 = label1.tolist()
        if isinstance(label2, torch.Tensor):
            label2 = label2.tolist()

        combined_labels= zip(label2,label1)
        newLabel = []
        for tup in combined_labels:
            if tup in self.index_map:
                newLabel.append(self.index_map[tup])
            elif (tup[1], tup[0]) in self.index_map:
                # Find the original index of the swapped tuple
                newLabel.append(self.index_map[(tup[1], tup[0])])
            else:
                # Handle the case where the tuple is not in index_map
                raise ValueError(f"Tuple {tup} not found in index_map either way")
        return newLabel
    
    def decode_labels(self, encoded_labels):
        """Returns the original tuples based on the index_map."""
        if isinstance(encoded_labels, torch.Tensor):
            encoded_labels = encoded_labels.tolist()
        
        num_labels = len(encoded_labels)
        # Pre-allocate lists with the correct size for efficiency
        first_labels = [None] * num_labels
        second_labels = [None] * num_labels

        # Fill both lists in a single pass
        for i, idx in enumerate(encoded_labels):
            label1, label2 = self.reverse_map[idx]  # Unpack tuple
            first_labels[i] = label1
            second_labels[i] = label2

        return first_labels, second_labels
    
    def get_number_probabilities(self, pred_prob_tensor):
        """
        Converts the probability tensor for encoded labels to a tensor of probabilities for each individual number.

        Parameters:
        - pred_prob_tensor: Tensor of shape (batch_size, encoded_label_space_size), representing
                            probabilities for each encoded label.
        - num_classes: Number of original classes (numbers) in the dataset.

        Returns:
        - Tensor of shape (batch_size, num_classes + 1) with the probability of each individual number.
        """
        batch_size = pred_prob_tensor.shape[0]
        number_probs = torch.zeros(batch_size, self.num_classes + 1)
        number_probs = number_probs.to(pred_prob_tensor.device)
        # Map each encoded label's probability to both of its numbers
        for encoded_label, (num1, num2) in self.reverse_map.items():
            number_probs[:, num1] += pred_prob_tensor[:, encoded_label]
            number_probs[:, num2] += pred_prob_tensor[:, encoded_label]

        return number_probs
    
    def create_ground_truth_tensors(self, label1 : list , label2 : list = None) -> torch.tensor:
        """
        Creates two ground truth tensors for the given (label1, label2) pair.
        If given only label1, it assumes it is encoded else it .

        For example, if the pair is (3,2):
        - Tensor A will have "1" at indices of pairs containing `label2` (2).
        - Tensor B will have "1" at indices of pairs containing `label1` (3).

        Returns:
        - gt_tensor_A: Tensor with 1s for pairs containing label2, else 0.
        - gt_tensor_B: Tensor with 1s for pairs containing label1, else 0.
        """
        if label2 == None:
            label1 , label2 = self.decode_labels(label1)
                                                 
        encodedLabel = zip(label1, label2)


        num_pairs = self.number_of_outputs(self.num_classes)-1 # -1 because of the fake,fake class
        #gives us a probability for each pair in the each of the labels of the batch
        gt_tensor_A = torch.zeros(len(label1) , num_pairs)
        gt_tensor_B = torch.zeros(len(label1) , num_pairs)

        for i, (lbl1, lbl2) in enumerate(encodedLabel):
            for (a, b), idx in self.index_map.items():
                if lbl1 in (a, b):
                    gt_tensor_A[i, idx] = 1  # Set to 1 if pair contains lbl2
                if lbl2 in (a, b):
                    gt_tensor_B[i, idx] = 1  # Set to 1 if pair contains lbl1

        return gt_tensor_A, gt_tensor_B

    def get_new_label_space(self):
        """Returns the new label space."""
        return list(self.new_label_space)
    def get_indexMap(self):
        """Returns the index map."""
        return self.index_map
    def get_reverseMap(self):
        """Returns the reverse map"""
        return self.reverse_map
    
    @staticmethod
    def number_of_outputs(num_classes):
        """Returns the number of possible outputs based on num_classes. assuming we will have 2 labels"""
        return (num_classes + 1) * (num_classes + 2) // 2


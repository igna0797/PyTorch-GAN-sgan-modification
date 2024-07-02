import argparse
import os
import numpy as np

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
    parser.add_argument("--max_lines", type=int, default=3, help="number of lines added as noise")
    parser.add_argument("--random_amount_lines", type=bool, default= False , help="if false always maximum amount")

# This is for loading a already done model
    parser.add_argument("-w ", "--weights_path", type=str, default="../../trainings/n-lineas_3_Random_False/generator_weights.pth", help="directory for the weigths of the generator")
    parser.add_argument("-o ", "--output_path", type=str, default="images/", help="directory for the Image returned by the generator")
    parser.add_argument("-s","--seed" ,type=int , default=3 ,help="seed for the generator for reproducible results")
    parser.add_argument("-i","--image_path" ,type=str , default="implementations/sgan/images/seed_1.000.png",help="directory for the image to discriminate")
    opt = parser.parse_known_args()[0]
    print(opt)
    return opt
def get_directory(__file__,max_lines=3 , random_amount_lines = False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the relative path to the pickle file
    directory = os.path.join(script_dir, "../../trainings/n-lineas_" + str(max_lines) + "_Random_"+ str(random_amount_lines))    
    #directory = "../../../content/drive/MyDrive/Redes neuronales/Monografia/n-lineas_" + str(opt.max_lines) + "_Random_"+ str(opt.random_amount_lines)
    return directory
def get_opt_path(__file__ , weights_path):
    abs_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path to the pickle file
    opt_path = os.path.join(abs_dir, os.path.dirname(weights_path), 'opt.pkl')
    #directory = "../../../content/drive/MyDrive/Redes neuronales/Monografia/n-lineas_" + str(opt.max_lines) + "_Random_"+ str(opt.random_amount_lines)
    return opt_path

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

import os
import numpy as np
import math
import pandas as pd  # Import Pandas for CSV handling
import pickle

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from utils import parseArguments , get_directory , get_opt_path , NoiseAdder

os.makedirs("images", exist_ok=True)
global cuda
cuda = True if torch.cuda.is_available() else False
print(f"Grapphics card accelertation: {cuda}")

if __name__ == "__main__":
    opt = parseArguments()
    # Get the directory where the script is located
    directory = get_directory(__file__, opt.max_lines , opt.random_amount_lines)
    optionsPath = os.path.join(directory,"opt.pkl")
    #Save options
    os.makedirs(os.path.dirname(optionsPath), exist_ok=True)  # Create the directory if it doesn't exist
    with open(optionsPath,"wb") as f:
       pickle.dump(opt,f)
else:
    #Load options    
    #directory = get_directory(__file__,3,False)
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
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.num_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.num_classes + 1), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label

if __name__ == "__main__":
  if opt.Training_output is not None:
      directory = opt.Training_output
  print("Los datos estan guardados en:" + directory)
  os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
  # Loss functions
  adversarial_loss = torch.nn.BCELoss()
  auxiliary_loss = torch.nn.CrossEntropyLoss()

  # Initialize generator and discriminator
  generator = Generator()
  discriminator = Discriminator()

  if cuda:
      generator.cuda()
      discriminator.cuda()
      adversarial_loss.cuda()
      auxiliary_loss.cuda()

  # Configure data loader
  os.makedirs("../../data/mnist", exist_ok=True)
  dataloader = torch.utils.data.DataLoader(
      datasets.MNIST(
          "../../data/mnist",
          train=True,
          download=True,
          transform=transforms.Compose(
              [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Lambda(lambda x: NoiseAdder.add_noise(x,opt)) ,transforms.Normalize([0.5], [0.5] )]
          ),
      ),
      batch_size=opt.batch_size,
      shuffle=True,
  )


  # Optimizers
  optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
  optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

  global FloatTensor
  global LongTensor
  FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
  LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

  """
  # Initialize weights
  if os.path.exists("/content/PyTorch-GAN-sgan-modification/implementations/sgan/generator_weights.pth") and os.path.exists("/content/PyTorch-GAN-sgan-modification/implementations/sgan/discriminator_weights.pth"):
    generator.load_state_dict(torch.load("/content/PyTorch-GAN-sgan-modification/implementations/sgan/generator_weights.pth"))
    discriminator.load_state_dict(torch.load("/content/PyTorch-GAN-sgan-modification/implementations/sgan/discriminator_weights.pth"))
    print("Loaded pre-trained weights.")
  else:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

  # Initialize weights
  if os.path.exists( directory + "/generator_weights.pth") and os.path.exists(directory +"/discriminator_weights.pth"):
      generator.load_state_dict(torch.load(directory + "/generator_weights.pth"))
      discriminator.load_state_dict(torch.load(directory +"/discriminator_weights.pth"))
      print("Loaded pre-trained weights.")
  else:
      generator.apply(weights_init_normal)
      discriminator.apply(weights_init_normal)
      print("Creating new weights.")
  """
# Create variables to track batch and epoch
  current_epoch = 0
  current_batch = 0

  # Check if a checkpoint file exists
  checkpoint_directory = os.path.join(directory , "checkpoint.pth")
  if os.path.exists(checkpoint_directory):
      checkpoint = torch.load(checkpoint_directory)
      current_epoch = checkpoint["epoch"]
      current_batch = checkpoint["batch"]
      generator.load_state_dict(checkpoint["generator_state_dict"])
      discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
      optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
      optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
      print(f"Loaded checkpoint from epoch {current_epoch}, batch {current_batch}")


  # Visualize a couple of real images from the dataset
  sample_data = next(iter(dataloader))
  sample_images, _ = sample_data

  # Save the visualization images in the same folder as generated images
  save_image(sample_images[:20], directory + "/dataset_visualization.png", nrow=5, normalize=True)

  # ----------
  #  Training
  # ----------
#gen_loss_log = open(directory + "/loss_log.txt", "w")

  # Initialize an empty list to collect loss data
  loss_data = []

  # Define the directory where you want to save images
  if opt.image_output is None:
    image_dir = directory + "/training/images"
  else:
      image_dir = opt.image_output
  # Create the directory if it doesn't exist
  os.makedirs(image_dir, exist_ok=True)
  for epoch in range(current_epoch, opt.n_epochs):
      for i, (imgs, labels) in enumerate(dataloader):
          current_batch = i
          current_epoch = epoch
          batch_size = imgs.shape[0]

          # Adversarial ground truths
          valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
          fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
          fake_aux_gt = Variable(LongTensor(batch_size).fill_(opt.num_classes), requires_grad=False)

          # Configure input
          real_imgs = Variable(imgs.type(FloatTensor))
          labels = Variable(labels.type(LongTensor))

          # -----------------
          #  Train Generator
          # -----------------

          optimizer_G.zero_grad()

          # Sample noise and labels as generator input
          z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

          # Generate a batch of images
          gen_imgs = generator(z)
          gen_imgs = NoiseAdder.add_noise(gen_imgs, opt)
          # Loss measures generator's ability to fool the discriminator
          validity, _ = discriminator(gen_imgs)
          g_loss = adversarial_loss(validity, valid)

          g_loss.backward()
          optimizer_G.step()

          # ---------------------
          #  Train Discriminator
          # ---------------------

          optimizer_D.zero_grad()

          # Loss for real images
          real_pred, real_aux = discriminator(real_imgs)
          d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

          # Loss for fake images
          fake_pred, fake_aux = discriminator(gen_imgs.detach())
          d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, fake_aux_gt)) / 2

          # Total discriminator loss
          d_loss = (d_real_loss + d_fake_loss) / 2

          # Calculate discriminator accuracy
          pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
          gt = np.concatenate([labels.data.cpu().numpy(), fake_aux_gt.data.cpu().numpy()], axis=0)
          d_acc = np.mean(np.argmax(pred, axis=1) == gt)

          d_loss.backward()
          optimizer_D.step()

          print(
              "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
              % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
          )

          batches_done = epoch * len(dataloader) + i
          if batches_done % opt.sample_interval == 0 and batches_done !=0:
              save_image(gen_imgs.data[:25], image_dir + "/%d.png" % batches_done, nrow=5, normalize=True)

              torch.save({
                "epoch": current_epoch,
                "batch": current_batch,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_G_state_dict": optimizer_G.state_dict(),
                "optimizer_D_state_dict": optimizer_D.state_dict(),
            }, checkpoint_directory)
              # Create a DataFrame from the collected data
              if os.path.exists(directory + "/training/loss_data.csv"):
                original_loss_df = pd.read_csv(directory + "/training/loss_data.csv")
                new_loss_df = pd.DataFrame(loss_data)
                # Create a list of DataFrames to concatenate
                dataframes_to_concat = [original_loss_df, new_loss_df]
                # Use pd.concat to concatenate the DataFrames
                loss_df = pd.concat(dataframes_to_concat)
              else:
                print("no enconto")
                loss_df = pd.DataFrame(loss_data)
              # Save the loss data to a CSV file
              loss_df.to_csv(directory + "/training/loss_data.csv", index=False)
              loss_data=[]
          loss_data.append({
              'Epoch': epoch,
              'Batch': i,
              'Discriminator Loss': d_loss.item(),
              'Discriminator Accuarcy' : 100 * d_acc,
              'Generator Loss': g_loss.item(),
          })

#loss_log.close()

  # Save generator weights
  torch.save(generator.state_dict(), directory + "/generator_weights.pth")
  # Save discriminator weights
  torch.save(discriminator.state_dict(), directory +"/discriminator_weights.pth")

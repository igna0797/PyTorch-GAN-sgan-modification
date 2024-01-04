<p align="center"><img src="assets/logo.png" width="480"\></p>

## PyTorch-GAN
Collection of PyTorch implementations of Generative Adversarial Network varieties presented in research papers. Model architectures will not always mirror the ones proposed in the papers, but I have chosen to focus on getting the core ideas covered instead of getting every layer configuration right. Contributions and suggestions of GANs to implement are very welcomed.

<b>See also:</b> [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN)

## Table of Contents
  * [Installation](#installation)
  * [Implementations](#implementations)
    + [Semi-Supervised GAN](#semi-supervised-gan)
      
## Installation
    $ git clone https://github.com/eriklindernoren/PyTorch-GAN
    $ cd PyTorch-GAN/
    $ sudo pip3 install -r requirements.txt

## Implementations   

### Semi-Supervised GAN
_Semi-Supervised Generative Adversarial Network_

#### Authors
Augustus Odena

#### Abstract
We extend Generative Adversarial Networks (GANs) to the semi-supervised context by forcing the discriminator network to output class labels. We train a generative model G and a discriminator D on a dataset with inputs belonging to one of N classes. At training time, D is made to predict which of N+1 classes the input belongs to, where an extra class is added to correspond to the outputs of G. We show that this method can be used to create a more data-efficient classifier and that it allows for generating higher quality samples than a regular GAN.

[[Paper]](https://arxiv.org/abs/1606.01583) [[Code]](implementations/sgan/sgan.py)

#### Run Example
```
$ cd implementations/sgan/
$ python3 sgan.py
```

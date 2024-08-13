<p align="center"><img src="assets/logo.png" width="480"\></p>

## PyTorch-GAN
Collection of PyTorch implementations of Generative Adversarial Network varieties presented in research papers. Model architectures will not always mirror the ones proposed in the papers, but I have chosen to focus on getting the core ideas covered instead of getting every layer configuration right. Contributions and suggestions of GANs to implement are very welcomed.

<b>See also:</b> [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN)

## Table of Contents
  * [Installation](#installation)
  * [Implementations](#implementations)
    + [Semi-Supervised GAN](#semi-supervised-gan)
      
## Installation
    $ git clone https://github.com/igna0797/PyTorch-GAN-sgan-modification.git
    $ cd PyTorch-GAN/
    $ sudo pip3 install -r requirements.txt

## Implementations   

### Semi-Supervised GAN
_Semi-Supervised Learning with Generative Adversarial Networks_

#### Authors
Ignacio Dominguez

#### Abstract
El presente trabajo se centra en el Aprendizaje Semi-Supervisado con Redes Generativas Adversariales (SGAN), una extensión de las Redes Generativas Adversariales.

El articulo, "Semi-Supervised Learning with Generative Adversarial Networks," de Augustus Odena, [[Paper]](https://arxiv.org/abs/1606.01583) [[Code]](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/sgan/sgan.py)
 sienta las bases de esta monografía. Esta investigación aborda la cuestión de cómo mejorar la eficiencia de un clasificador en un entorno de aprendizaje semi-supervisado, donde las etiquetas de clase disponibles son escasas.

En particular, el trabajo propone modificaciones a un generador y un clasificador en un marco de Generative Adversarial Networks (GANs) semi-supervisado, conocido como SGAN. La novedad radica en que el discriminador de la GAN es modificado para realizar tareas de clasificación con  multiples salidas, no solo de  verdadero-falso. Esto no solo mejora el rendimiento de la clasificación en tareas semi-supervisadas, sino que también da lugar a la generación de datos de alta calidad.

En esta monografía, se explorarán las bases teóricas de las redes Convolucionales (ConvNets) y las GANs, así como las modificaciones realizadas en el modelo SGAN. Se describirá en detalle cómo estas ideas fundamentales se combinan para lograr una mayor eficiencia en tareas de clasificación semi-supervisada y una mejora en la calidad de los datos generados.

Además, se presentarán las pruebas extras  realizadas como parte de este trabajo, donde se ha agregado ruido controlado al modelo y se ha observado su impacto en el rendimiento y la generación de datos.

[[Paper]](Monografia_Redes_Neuronales.pdf) [[Code]](implementations/sgan/sgan.py)

#### Run Example

You can download just Run-Notebook.ipynb to see an example beeing used 

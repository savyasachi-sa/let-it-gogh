# let-it-gogh

Description
===========
This is project on Neural Style Transfer developed by team *Untitled1* composed of Ria Aggarwal, Sandalika Sapra & Savyasachi.[UCSD ECE-285 Fall2019 Project]


Code Organization
=================

There are two main directories - 
 * *neural-style-transfer* - holds the code for the implementation of "A Neural Algorithm of Artistic Style" by Gatys
 * *cycleGan* - holds the code for implementation of style transfer using cycle GANs

### neural-style-transfer:

* *demo.ipnyb* - Run a demo for style-transfer implementation.
* *StyleTransferDataset.py* - Dataset Class for managing data for style transfer. It loads image pairs for style & content and takes care of pre-processing. It supports different file-size and initializations. See class documentation for more details.
 * *StyleTransferNet.py* - Holds the main neural-net and loss function confugurations etc. Has an init() method which takes layer and weight configurations and creates and configures the net, loss functions and targets. 
* *StyleTransferTrain.py* - Holds the code for training. Has a train method which takes layer configurations, a dataset entry and returns the output image after training for the desired number of iterations.  
* *Experiments/*
  * *Layers.ipnyb* - Experiments over considering different layers for Style & Content (Reproduces Fig 4 of the report)
  * *Initialization.ipnyb* - Experiments over how Initialization effects the output image (Reproduces Fig 3 of the report)
* Images/ - Holds the images used for training and demo purposes. Contains two directories for Style & Content respectively. Files with the same name from Style and Content are picked as pair for style transfer.


### cycleGAN:

* *demo_cycleGAN.ipynb* - Run a demo of CycleGAN implementation on pretrained models from created checkpoints.
* *train_cycleGAN.ipynb* - Train a new cycleGAN model.
* *experiments/*
  * *cycleGANexp.ipynb* - Trains similar cycleGAN architecture. Tracks adversarial, cyclic, identity and discrimiinator losses to produce Fig 6 of the report. Weights to losses experimented with to produce Fig 8 and 9 of the report. Noticeable variations in results with initialization.
* *train.py* - Creates network architecture, defines optimizers, training method and losses, and saves checkpoints to directory.
* *test.py* - Testing module that loads trained checkpoints into the defined network architecture for style transfer on test data, results of which are saved to an output directory.
* *main.py* - Module to utilize arguments from the command line for parameter and experiment setup and run.
* *architecture/*
  * *discriminators.py* - Contains discriminator architecture (convnet)
  * *generators.py* - Contains generator architecture (conv and residual layers)
  * *options.py* - Defines layer and initialization types used to create architectures in generators.py and discriminators.py
* *utils.py* - Contains code for image pool strategy and the learning rate decay

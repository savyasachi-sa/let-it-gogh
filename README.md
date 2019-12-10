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

* *demo_cycleGAN.ipynb* - notebook for the demo of CycleGAN implementatiom.
* *train_cycleGAN.ipynb* - notebook for training the model.
* *experiments* - contains all the experimental notebooks for cycleGANs
* *train.py* - training module
* *test.py* - testing module
* *main.py* - module for invoking testing/training 
* *architecture* - contains the architecture of Generators and Discriminators


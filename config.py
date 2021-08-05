import numpy as np 
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision import transforms
import matplotlib.pyplot as plt 

CPKT = 'checkpoints/'
PLOTS = 'plots/'
ROOT = './data'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D_STEPS_PER_ITER = 5                # Number of discriminator updates per iteration
G_STEPS_PER_ITER = 5                # Number of Generator updates per iteration

PLOT_NUM = 9                        # Number of images to plot
LR_G = 1e-4                         # Learning rate for generator
LR_D = 4e-4                         # Learning rate for generator
LR_C = 3e-4                         # Learning rate for classifier
BATCH_SIZE = 256                    # Batch size
LATENT_DIM = 100                    # Latent dimension of noise vector
EPOCHS = 100                        # Epochs for GANs
C_EPOCHS = 10                       # Epochs for classifier    
G_CHANNELS = 256                    # Generator input channels for conv layer after linear layer
DROPOUT = 0.3                       # Dropout rate for discriminator
PRINT_FREQ = 100                    # Print after this number of iterations
INTERPOLATE_STEPS = 10              # Number of steps to interpolate between 2 points while traversing in latent dimension
NROW = 10                           # Number of images to print in row
NUM_SAMPLES = 20                    # Number of images to generate for classifier

######### FLAGS ##########
G_SPECTRAL_NORM = True              # Spectral norm in generator
D_SPECTRAL_NORM = True              # Spectral norm in discriminator
BATCH_NORM = True                   # Batch Norm flag
BOOL_PLOT = True                    # Plot images while training
TRAIN_CLASSIFIER = True             # Trains the classifier model [Better to train GANs first then train classifier]
PRED_ON_GAN_OUTPUT = True           # Making prediction on images generated from GANs using classifier model
PLOT_DISTRIBUTION = False           # Plot distribution of generated images over classes
PLOT_20 = True                      # Plot 20 random images from generator with predictions
SELF_ATTENTION = True               # Apply Self Attention in Generator and Discriminator before Last conv. layer

GET_NOISE = lambda x : torch.rand(size = (x, LATENT_DIM), device = DEVICE)

TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),      
    transforms.Normalize((0.5,), (0.5,))
])

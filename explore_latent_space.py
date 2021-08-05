import numpy as np
from generator import *
from discriminator import *
from config import *
from dataset import *
from model import *

def interpolate(p0, p1, n_steps=10):
    
    ratios = np.linspace(0, 1, num = n_steps)
    vectors = torch.tensor([], device=DEVICE)
    for ratio in ratios:
        vector = (1.0 - ratio) * p0 + ratio * p1
        vectors = torch.cat((vectors, vector))
    
    return vectors


generator = Generator(latent_dim = LATENT_DIM, g_channels = G_CHANNELS,
                             batch_norm=BATCH_NORM, g_spectral_norm=G_SPECTRAL_NORM)

# Loading best generator model over EPOCHS
generator.load_state_dict(torch.load(CPKT+f"GANS_EPOCH_{EPOCHS}.pth"))

discriminator = Discriminator(d_spectral_norm=D_SPECTRAL_NORM)

generator = generator.to(DEVICE)
discriminator = discriminator.to(DEVICE)

model = Model(
    latent_dim = LATENT_DIM,
    dataloader = train_dataloader,
    generator = generator, 
    discriminator = discriminator,
    batch_size = BATCH_SIZE,
    lr_g = LR_G,
    lr_d = LR_D)

# Taking two random points in latent space
p0, p1 = GET_NOISE(1), GET_NOISE(1)
n_steps = INTERPOLATE_STEPS

# Interpolate and generate latent vectors between those two points
latent_vecs = interpolate(p0, p1, n_steps=n_steps)

# Generate images using these latent vectors
model.generator.eval()
model.plot_images(latent_vec = latent_vecs, num = 1, filename = f'figures/LATENT_NSTEPS_{n_steps}.png')


import numpy as np
from generator import *
from discriminator import *
from config import *
from dataset import *
from model import *
import matplotlib.pyplot as plt 

generator = Generator(latent_dim = LATENT_DIM, g_channels = G_CHANNELS,
                             batch_norm=BATCH_NORM, g_spectral_norm=G_SPECTRAL_NORM)

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
    lr_d = LR_D
    )

G_losss = []
D_loss_rs = []
D_loss_fs = []
D_accs = []

print('Training GANs...')
for epoch in range(EPOCHS):
    print("EPOCH: ", epoch+1)

    G_loss, (D_loss_r, D_loss_f), D_acc = model.train_epoch(print_freq = PRINT_FREQ, epoch = epoch)
    G_losss.append(G_loss)
    D_loss_rs.append(D_loss_r)
    D_loss_fs.append(D_loss_f)
    D_accs.append(D_acc)

torch.save(model.generator.state_dict(), CPKT+f"GANS_EPOCH_{EPOCHS}.pth")

model.plot_images(num = 9, filename = f"figures/PLOT_after_{EPOCHS}_EPOCHS.png")

plt.plot(G_losss, label='Gen_Loss')
plt.plot(D_loss_rs, label='D_loss_R')
plt.plot(D_loss_fs, label='D_loss_F')
plt.legend()
plt.savefig("figures/GANs_Loss_curves.png")
plt.close()

plt.plot(D_accs, label='D_accuracy_on_Generator_output')
plt.legend()
plt.savefig("figures/Discriminator_acc.png")
plt.close()



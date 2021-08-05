from config import *
'''
    Defining Train/Valid/test dataloaders
'''
dataset = torchvision.datasets.MNIST(root = ROOT, train = True, transform = TRANSFORM, download = True)
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [50000, 10000])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 8, drop_last=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 8, drop_last=True)

test_dataset = torchvision.datasets.MNIST(root = ROOT, train = False, transform = TRANSFORM, download = True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 8, drop_last=True)


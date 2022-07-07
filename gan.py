import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from time import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Count of using GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device())

'''
Comment: 
Is there any reason to choose the last transform as normalize([0.5], [1])?
Normally, people set it as normalize([0.5], [0.5]) to make the scale of input image as [-1, 1] from [0, 1].
FYI) To make the scale of RGB image as [-1, -1] from [0, 1], you can use normalize([0.5, 0.5, 0.5], [0.5,0.5,0.5]).
'''
'''
I miscalculated :(
'''
transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

latent_size = 64
batch_size = 128

'''
Comment: Good. Large enough.
'''
total_epoch = 150
learning_rate = 0.0001

data_root = './data'
train_data = datasets.MNIST(data_root, transform=transform, train=True, download=True)

dataloader = DataLoader(train_data, batch_size = batch_size, shuffle=True, num_workers = 4)

# example data
train_x, train_y = next(iter(dataloader))
img = train_x[0].squeeze()
# label = train_y[0]
# plt.imshow(img, cmap="gray")
# plt.title = label
# plt.show()

'''
Comment:
Do you know why we are using "Tanh" as the last activation of the generator?
Also, guess the reason for the "sigmoid" which is the last activation of the discriminator.
'''
'''
Tanh : generator should generate pixel values that range from -1 to 1 as in our train_data with transform parameter
sigmoid : discriminator should output values close to 0 for fake images and 1 for real images so sigmoid is used
'''
# https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/
# Leaky ReLU for discriminator, relu&tanh for generator
class Generator(nn.Module):
    def __init__(self, latent_size, image_size):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(inplace = True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, image_size),
            nn.Tanh()
        ) 
    def forward(self, z):
        output = self.layer(z)
        return output

'''
Comment:
Maybe, you cannot use the argument "inplace" for relu with higher version of pytorch.
I remember that this is for memory control of older version of pytorch, but not sure.
You can search about it.
'''
'''
I wasn't able to find such articles so I hope it isn't problematic.
I googled it, and took a look in to official documentation of pytorch 1.12 and it seemed fine!
'''
'''
Good. I mean "higher version of pytorch" around 1.7-1.8, and it is changed again...
'''
class Discriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        output = self.layer(x)
        return output

discriminator = Discriminator(len(img.flatten())).to(device)
discriminator = nn.DataParallel(discriminator, output_device=[0,1])
generator = Generator(latent_size, len(img.flatten())).to(device)
generator = nn.DataParallel(generator, output_device=[0,1])
dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

criterion = nn.BCELoss().to(device)
average_time = 0
for epoch in range(total_epoch):
    start_time = time()
    for i, (image, label) in enumerate(dataloader):
        
        real_image = image.view((image.size(0), -1)).to(device)
        ones = torch.ones((image.size(0), 1)).to(device)
        zeros = torch.zeros((image.size(0), 1)).to(device)
        # train discriminator
        dis_optimizer.zero_grad()
        real_output = discriminator(real_image)
        z = torch.randn((image.size(0), latent_size)).to(device)
        fake_image = generator(z)
        '''
        Comment:
        To update discriminator, you don't have to save gradient of generator.
        Thus, you may use "detach()" to prevent additional gradient flow to generator.
        ex) fake_output = discriminator(fake_image.detach())

        Be careful. This is not true for updating generator.
        In that case, the gradient of discriminator is required.        
        
        For detailed information, please google it.
        '''
        '''
        Based on my code, generator gets updated in 'train discriminator' phase which is problematic right?
        I thought we only give discriminator.parameters() as parameter to dis_optimizer so it wouldn't be problematic.
        Also, it is not the same in updating generator since we only update 
        '''
        '''
        Checked! Totally fine without detach(), but just for clean code.
        '''
        # fake_output = discriminator(fake_image.detach())
        fake_output = discriminator(fake_image)

        real_loss = criterion(real_output, ones)
        fake_loss = criterion(fake_output, zeros)
        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        dis_optimizer.step()

        # train generator
        gen_optimizer.zero_grad()
        z = torch.randn((image.size(0), latent_size)).to(device)
        fake_image = generator(z)
        fake_output = discriminator(fake_image)
        generator_loss = criterion(fake_output, ones)

        generator_loss.backward()
        gen_optimizer.step()
    if epoch == 0:
        real_image = real_image.view(real_image.size(0), 1, 28, 28)
        save_image(real_image[:25], "./result/real.png", nrow=5, normalize=True)
    if epoch % 10 == 0:
        fake_image = fake_image.view(fake_image.size(0), 1, 28, 28)
        # dir = f"./result/detach"
        dir = f"./result/no_detach"
        if not os.path.exists(dir):
            os.makedirs(dir)
        save_image(fake_image, os.path.join(dir, f"{epoch}.png"), normalize=True)
    t = time()-start_time
    average_time += t
    print(f'Epoch {epoch}/{total_epoch} || discriminator loss={discriminator_loss:.4f}  || generator loss={generator_loss:.4f} || time {t:.3f}')
torch.save(discriminator.state_dict(), os.path.join(dir,"discriminator.ckpt"))
torch.save(generator.state_dict(), os.path.join(dir,"generator.ckpt"))
print(average_time/epoch)

# experiment
# does order matter? g then d // d then g -> in the paper, d->g
# how about cnn rather than fcnn?
# learning rate : 0.001 -> overshoot
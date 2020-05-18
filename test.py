
%matplotlib inline

#GAN과 DCGAN에 대하여: https://dreamgonfly.github.io/2018/03/17/gan-explained.html

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchsummary import summary
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
]) #transform 정의

batch_size = 128 #batch size: 128
z_dim = 100 # num_latent_variable


database = dataset.MNIST('mnist', train = True, download = True, transform = transform)
#Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
#minst dataset download&transform
train_loader = torch.utils.data.DataLoader(
    #dataset.MNIST('mnist', train = True, download = True, transform = transform),
    database,
    batch_size = batch_size,
    shuffle = True
)

def weights_init(m): #가중치 초기화 함수 정의
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #gpu or cpu 사용

class Generator_model(nn.Module): #generator class 정의
    def __init__(self, z_dim):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256 * 7 * 7) #generator 모델 구성(각 유닛 정의)
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh() # 101-103p 참조
        )
    def forward(self, input): #forward method 정의
        x = self.fc(input)
        x = x.view(-1, 256, 7, 7)
        return self.gen(x)

generator = Generator_model(z_dim).to(device) #generator 모델을 위한 object 만들기

generator.apply(weights_init) #가중치 초기화

summary(generator, (100, )) #generator summary 출력

class Discriminator_model(nn.Module): #discriminator class 정의
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential( #generator 모델 구성(각 유닛 정의)
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01)
        )
        self.fc = nn.Linear(2048, 1)
    def forward(self, input): #forward method 정의
        x = self.disc(input)
        return F.sigmoid(self.fc(x.view(-1, 2048))) #sigmoid 적용

discriminator = Discriminator_model().to(device) #create discriminator object

discriminator.apply(weights_init) #가중치 초기화

criterion = nn.BCELoss() #손실 함수 정의

# create a batch (whose size 64) of fixed noise vectors (z_dim=100)
fixed_noise = torch.randn(64, z_dim, device=device)  #노이즈 만들기 dimension:100

doptimizer = optim.Adam(discriminator.parameters())
goptimizer = optim.Adam(generator.parameters()) #adam optimizer 사용

real_label, fake_label = 1, 0 #discriminator label(real: 1, fake:0의 값)

image_list = []
g_losses = []
d_losses = [] #training 결과 저장할 metrics
iterations = 0
num_epochs = 50

for epoch in range(num_epochs): #training 시작
    print(f'Epoch : | {epoch + 1:03} / {num_epochs:03} |')
    for i, data in enumerate(train_loader):

        discriminator.zero_grad() #discriminator의 gradient 0으로 초기화

        real_images = data[0].to(device) #이미지 불러오기

        size = real_images.size(0)
        print("for each batch, size =" + size)
        label = torch.full((size,), real_label, device=device) # 이미지로부터 label 생성
        print("for each batch, label =" + label)
        d_output = discriminator(real_images).view(-1) #discriminator output
        print("for each batch, d_output (real) =" + d_output)
        derror_real = criterion(d_output, label) #discriminator error 계산

        print("for each batch, derror_real =" + derror_real)
        derror_real.backward() #gradient 계산

        noise = torch.randn(size, z_dim, device=device) #노이즈 생성
        fake_images = generator(noise) #노이즈를 generator에 넣어서 fake images 생성
        label.fill_(0)  # _: in-place-operation / generated images로부터 label 생성
        d_output = discriminator(fake_images.detach()).view(-1) #discriminator에 fake images 넣기
        print("for each batch, d_output(fake) =" + d_output)
        derror_fake = criterion(d_output, label) #fake image의 discriminator error 계산
        print("for each batch, derror_fake =" + derror_fake)
        derror_fake.backward()  #~의 gradient 계산

        derror_total = derror_real + derror_fake # total discriminator error
        doptimizer.step() #update discriminator weight

        generator.zero_grad() #generator의 gradient 0으로 초기화
        label.fill_(real_images)  # _: in-place-operation; the same as label.fill_(1) #label real(1)로
        d_output = discriminator(fake_images).view(-1) #discriminator output으로...
        gerror = criterion(d_output, label) #generator error 계산
        gerror.backward() #gradient 계산

        goptimizer.step() #update generator weight

        if i % 50 == 0: #save losses
            print(
                f'| {i:03} / {len(train_loader):03} | G Loss: {gerror.item():.3f} | D Loss: {derror_total.item():.3f} |')
            g_losses.append(gerror.item())
            d_losses.append(derror_total.item())

        if (iterations % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(train_loader) - 1)):
            with torch.no_grad(): #save noise images
                fake_images = generator(fixed_noise).detach().cpu()
            image_list.append(vutils.make_grid(fake_images, padding=2, normalize=True))

        iterations += 1

plt.figure(figsize=(10,5)) #training 동안 generator와 discriminator의 loss 그래프 그리기
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_losses,label="Generator")
plt.plot(d_losses,label="Discriminator")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

for image in image_list:
    plt.imshow(np.transpose(image,(1,2,0))) #image_list의 image iterate
    plt.show()

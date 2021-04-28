import torch
import torchvision
import torchvision.transforms as transforms
import ssl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
ssl._create_default_https_context = ssl._create_unverified_context

transform_set = [
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomVerticalFlip(p=0.5),
     #transforms.RandomAffine(degrees=(-30,30), translate=(0, 0.5), scale=(0.4, 0.5), shear=(0,0), fillcolor=(0,255,255)),
     transforms.RandomGrayscale(p=0.5),
     #transforms.ColorJitter()
]


transform = transforms.Compose(
    [
     transforms.Resize(64),
     transforms.RandomChoice(transform_set),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])


# Parameters
params = {'batch_size': 220,
          'shuffle': True,
          'num_workers': 16}


#trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                        download=True, transform=transform)
trainset = torchvision.datasets.ImageFolder(root="../food11re/food11re/skewed_training", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, **params)

#testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                       download=True, transform=transform)

testset = torchvision.datasets.ImageFolder(root="../food11re/food11re/evaluation", transform=transform)
testloader = torch.utils.data.DataLoader(testset, **params)






import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 25, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(25, 25, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(25, 25, 3)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(25, 25, 3)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(100, 50)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(50, 25)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(25, 11)
        self.relu5 = nn.ReLU()


    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = self.conv3(y)
        y = self.relu3(y)
        y = self.pool3(y)
        y = self.conv4(y)
        y = self.relu4(y)
        y = self.pool4(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y 

if __name__ == "__main__":
    import os.path
    from os import path
    PATH = './cifar_net.pth'
    if True:
        print("Training")
        net = Net()
        net.to(device)
        net.load_state_dict(torch.load(PATH))
        import torch.optim as optim
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.99)
        for epoch in range(25):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    
                # print statistics
                running_loss += loss.item()
                if i % 20 == 19:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                    running_loss = 0.0
        print('Finished Training')
        torch.save(net.state_dict(), PATH)
    
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    
    # print images
    #imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(4)))
    net = Net()
    net.load_state_dict(torch.load(PATH))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % predicted[j] for j in range(4)))
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    

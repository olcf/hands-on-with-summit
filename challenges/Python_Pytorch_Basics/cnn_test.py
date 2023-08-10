import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import time as tp

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)            # -> n, 400
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x
    


def train_and_test_cnn(num_epochs: int, batch_size: int, device: str, learning_rate: float,
                       transform: torchvision.transforms.Compose, data_root: str) -> [float, float]:
    # CIFAR10 Datasets
    train_dataset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform)

    # Data Loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model, Loss, and Optimizer
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print('')
    print('Number of epochs:', num_epochs)
    print('Batch size:', batch_size)
    print('Learning rate:', learning_rate)
    print('')

# Training Loop
    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = tp.time()

    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 500 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    if device == 'cuda':
        torch.cuda.synchronize()
    training_time = tp.time() - start_time
    print('Finished Training')

# Testing Loop
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

    accuracy = 100.0 * n_correct / n_samples
    

    print('')
    print(f'Accuracy of the network: {accuracy} %')
    print('Execution time of training loop: ', training_time, 's')

    print('')
    if ( (accuracy >= 60.0) and (learning_rate==0.001)):
        print('Success!')
    elif ( (accuracy >= 60.0) and (learning_rate!=0.001)):
        print('Great! But change your learning rate back to 0.001 for the challenge')
    else:
        print('Accuracy not 60% or above, try again!')


    return accuracy, training_time



####### TESTS #######
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
device = 'cpu' # interestingly cpu is faster than mps
data_root = '/Users/kevinpyx/Documents/Datasets/CV/The CIFAR-10 dataset'
learning_rate = 0.001

epoch_list = [2, 4, 6, 8, 10, 12, 14, 16]
batch_size_list = [1, 2, 4, 5, 10, 20, 25, 40, 50]

acc_time_list = np.zeros((len(epoch_list), len(batch_size_list), 2))

for i, epoch in enumerate(epoch_list):
    for j, batch_size in enumerate(batch_size_list):
        acc, time = train_and_test_cnn(epoch, batch_size, device, learning_rate, transform, data_root)
        acc_time_list[i,j,0] = acc
        acc_time_list[i,j,1] = time
        print('')
        print('Epoch:' + str(epoch) + ', Batch size:' + str(batch_size))
        print('Accuracy:', acc)
        print('Time:', time)
        print('')

# print summary
print('Summary')
for i, epoch in enumerate(epoch_list):
    for j, batch_size in enumerate(batch_size_list):
        print('Epoch:' + str(epoch) + ', Batch size:' + str(batch_size))
        print('Accuracy:', acc_time_list[i,j,0])
        print('Time:', acc_time_list[i,j,1])
        print('')

# Plot1: Accuracy vs Epoch
# Every batch size is one line of different color
# Epoch number is the x axis
# Accuracy is the y axis
plt.figure()
for i, batch_size in enumerate(batch_size_list):
    plt.plot(epoch_list, acc_time_list[:,i,0], label='Batch size: ' + str(batch_size))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Plot2: Time vs Epoch
# Every batch size is one line of different color
# Epoch number is the x axis
# Time is the y axis
plt.figure()
for i, batch_size in enumerate(batch_size_list):
    plt.plot(epoch_list, acc_time_list[:,i,1], label='Batch size: ' + str(batch_size))
plt.xlabel('Epoch')
plt.ylabel('Time')
plt.legend()
plt.show()


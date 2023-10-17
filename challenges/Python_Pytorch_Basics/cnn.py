# cnn.py
# Author: Michael A. Sandoval
# Adapted from Patrick Loeber's CNN tutorial https://github.com/python-engineer/pytorchTutorial
# Trains a Convolutional Neural Network (CNN) on the CIFAR10 dataset using PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import time as tp

############## FUNCTION DEFINITIONS ######################

def imshow_grid(classes, imgs, labels, predics, num_imgs):
    '''Create a grid of a batch of images using imshow.
    The prediction is compared to the real classification
    in the title. Number of grid rows and columns determined
    from batch size (num_imgs)'''

    # Find number rows and columns
    cols = int( np.floor( np.sqrt(num_imgs) ) )
    rows = int( np.ceil( num_imgs / cols ) )

    # Create figure
    figure, ax = plt.subplots(rows, cols, constrained_layout=True, squeeze=False)

    for i in range(0, num_imgs):

        # Extract a single image and renormalize the colors
        single_img = imgs[i,:,:,:]
        single_img = single_img / 2 + 0.5  # unnormalize

        # Find which column and row we are in
        col_ind = i % cols
        row_ind = i // cols

        # Plot the images (also turn off axis labels, set the title)
        ax[row_ind,col_ind].set_title(classes[ labels[ i ] ] + ', Guess: ' + classes[ predics[ i ] ], fontsize=7)
        ax[row_ind,col_ind].axis("off")
        ax[row_ind,col_ind].imshow(single_img.permute(1, 2, 0)) # need to re-order the dimensions to plot correctly

    # Turn off axes for remaining subplots if number of images didn't span entire grid
    remaining = (cols * rows) - num_imgs
    for i in range(1, remaining + 1):
        ax[-1, 0 - i].axis("off")

    # Save Plot
    figure.savefig('last_batch.png')


def overall_results(classes, num_correct, num_samples, num_predictions, acc_network, batches, epochs):
    '''Create a side-by-side bar graph of the success rate of identifying
    a given class and the success rate of a given prediction'''

    # Create figure
    figure, ax = plt.subplots()

    # Accuracy Data
    acc_class = 100.0 * np.array(num_correct) / np.array(num_samples)
    acc_pred = 100.0 * np.array(num_correct) / np.array(num_predictions)

    # Set x-axis spacing
    ind = np.linspace(0,30,10) # 10 classes so need 10 entries
    barWidth = 0.85

    # Create green Bars
    ax.bar(ind - barWidth/2, acc_class, color='#b5ffb9', edgecolor='white', width=barWidth, label='Identification Success (%)')
    # Create orange Bars
    ax.bar(ind + barWidth/2, acc_pred, color='#f9bc86', width=barWidth, label='Prediction Success (%)')
    # Create horizontal dashed line
    ax.axhline(acc_network, 0, 1, color='black', ls='--', lw=0.75, label='Overall Network Accuracy (%)')

    # Custom Axes
    ax.set_xticks(ind, classes)
    ax.set_ylim(0.0,100.0)
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    # Title and Legend
    ax.set_title('Overall Results for: batches = %s, epochs = %s' %(batches, epochs))
    ax.legend()

    # Save Plot
    figure.savefig('overall_results.png')


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

############## END OF FUNCTION DEFINITIONS #################



#################### CNN WORKFLOW ##########################

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('The device you are using is: ',device)

# Hyper-parameters 
num_epochs = 4 # CHANGE-ME
batch_size = 4 # CHANGE-ME
learning_rate = 0.001

print('')
print('Batch size:', batch_size)
print('Number of epochs:', num_epochs)
print('Learning rate:', learning_rate)
print('')

# First transform image to tensor
# Then normalize tensor to range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('Plane', 'Car', 'Bird', 'Cat',
           'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

torch.cuda.synchronize()
t1=tp.time()

# Training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 500 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

torch.cuda.synchronize()
t2=tp.time()
print('Finished Training')

# Testing Loop
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    n_class_predics = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # torch.max returns (values , prediction indices)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        # For a batch, compare image predictions to real labels
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
            n_class_predics[pred] += 1

#################### END OF CNN WORKFLOW #####################



#################### ANALYSIS / STATS ########################

# E.g., Success rate of IDENTIFYING frogs
print('')
print('Accuracy of Class Samples (e.g., Number of Frogs Correct / Number of Frog Samples)')
print('==================================================================================')
for i in range(10):
    acc_class = 100.0 * n_class_correct[i] / n_class_samples[i]
    print(f'Accuracy of {classes[i]} samples: {acc_class} %')

# E.g., Success rate when GUESSING "frog"
print('')
print('Accuracy of Predictions (e.g., Number of Frogs Correct / Number of Frog Predictions)')
print('====================================================================================')
for i in range(10):    
    if n_class_predics[i] != 0:
        acc_pred = 100.0 * n_class_correct[i] / n_class_predics[i]
        print(f'Accuracy of {classes[i]} predictions: {acc_pred} %')
    else:
        acc_pred = 0.0
        print(f'No {classes[i]} predictions were made')

# Overall Network Accuracy
acc = 100.0 * n_correct / n_samples
print('')
print(f'Accuracy of the network: {acc} %')
print('Execution time of training loop: ', t2-t1, 's')

print('')
if ( (acc >= 60.0) and (learning_rate==0.001)):
    print('Success!')
elif ( (acc >= 60.0) and (learning_rate!=0.001)):
    print('Great! But change your learning rate back to 0.001 for the challenge')
else:
    print('Accuracy not 60% or above, try again!')

# Get the last batch of images
dataiter = iter(test_loader)
images, labels = next(dataiter)
for images, labels in dataiter:
    pass

# Call plot functions
plt.rc('font', family='serif') # set plot font style

imshow_grid(classes= classes, imgs= images, labels= labels, predics= predicted, num_imgs= batch_size)

overall_results(classes= classes, num_correct= n_class_correct, num_samples= n_class_samples, \
                num_predictions= n_class_predics, acc_network= acc, batches= batch_size, epochs= num_epochs)

#################### END OF ANALYSIS / STATS ###################

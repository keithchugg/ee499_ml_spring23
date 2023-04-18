# Import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt

# Parameters for the model
n_pixels = 28 * 28
n_classes = 10

# Parameters for the training
USE_CPU = False
reg_val = 1e-4
lr = 0.001 / 2

# drate = 0.25 in classifier and lr = 0.001 / 2 yields 94+ % on val. in 20 epochs. batch = 128
# drate = 0.35 in classifier and lr = 0.001 / 3 yields ~93.6 % on val. in 20 epochs. batch = 256
# drate = 0.5 in classifier and lr = 0.001 / 4 yields ~93.6 % on val. in 20 epochs. batch = 256


# All datasets are subclasses of torch.utils.data.Dataset i.e, they have __getitem__ and __len__ methods implemented
from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]) # transforms.ToTensor() converts the image to a tensor and transforms.Normalize() normalizes the tensor
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)

image, label = trainset[0] 
print(image.shape) # torch.Size([1, 28, 28])
print(label) 
input_image_shape = image.shape

trainset, valset = torch.utils.data.random_split(trainset, [50000, 10000])
# Final sizes are 50000, 10000, 10000
print(f'Train set size: {len(trainset)}, Validation set size: {len(valset)}, Test set size: {len(testset)}')

class FullDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset        
        self.data, self.labels = self._load_data()
        
    def _load_data(self):
        data = []
        labels = []
        
        for i in range(len(self.dataset)):
            x, y = self.dataset[i]
            data.append(x)
            labels.append(y)
        
        return torch.stack(data), torch.tensor(labels)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
        
trainset = FullDataset(trainset)
valset = FullDataset(valset)

batchsize = 256
# Shuffle the data at the start of each epoch (only useful for training set)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)
train_eval_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=False)
valloader = torch.utils.data.DataLoader(valset, batch_size=batchsize, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False)

# Define the model
# for an alternative approach, see: mini_vgg_like_torch_1.py
def vgg_like_block(in_channels, num_filters, ksize=3, drate=0.25, pad='same'):
    return nn.Sequential(
        nn.Conv2d(in_channels, num_filters, kernel_size=ksize, padding=pad),
        nn.ReLU(),
        nn.BatchNorm2d(num_filters),
        nn.Conv2d(num_filters, num_filters, kernel_size=ksize, padding=pad),
        nn.ReLU(),
        nn.BatchNorm2d(num_filters),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout2d(p=drate)
    )

def classifier_mlp(n_in, n_hidden, n_classes, drate=0.25):
    return nn.Sequential(
        nn.Linear(n_in, n_hidden),
        nn.ReLU(),
        nn.BatchNorm1d(n_hidden),
        nn.Dropout1d(p=drate),
        nn.Linear(n_hidden, n_classes)
    )

class MiniVgg(nn.Module):
    def __init__(self, b1_filters=32, b2_filters=64, H=28, fc_nodes=512, n_classes=10):
        super().__init__()
        self.block1 = vgg_like_block(1, b1_filters)
        self.block2 = vgg_like_block(b1_filters, b2_filters)
        assert H % 4 == 0, f'the image height and width must be a multiple of 4: you passed H = {H}'
        mlp_in_size = (H * H // 16) * b2_filters  # the H and W are both reduced by 4 with 2 max-pool layers.
        self.classifier = classifier_mlp(mlp_in_size, fc_nodes, n_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        y = self.block1(x)
        y = self.block2(y)
        y = y.view(batch_size, -1)
        y = self.classifier(y)
        return y

model = MiniVgg()
print(model)
summary(model, input_image_shape)  # call summary before moving the model to a device...

criterion = nn.CrossEntropyLoss() # includes softmax (for numerical stability)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg_val)  # default learning rate is 0.001

# set the device to use and move model to device

if USE_CPU:
    device = torch.device("cpu")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.torch.backends.mps.is_available():
    device = torch.device("mps") # MPS acceleration is available on MacOS 12.3+
else:
    device = torch.device("cpu")

print(f'Using device: {device}')
model.to(device) # Move model to device


# Define function to call for each training epoch (one complete pass over the training set)
def train(model, trainloader, criterion, optimizer, device):
    model.train() # set model to training mode
    running_loss = 0; running_acc = 0
    with tqdm(total=len(trainloader), desc=f"Train", unit="batch") as pbar:
        for n_batch, (images, labels) in enumerate(trainloader): # Iterate over batches
            images, labels = images.to(device), labels.to(device) # Move batch to device
            optimizer.zero_grad()
            output = model(images) # Forward pass
            loss = criterion(output, labels) # Compute loss
            loss.backward() # Backward pass
            optimizer.step() # Update weights
            running_loss += loss.item()
            running_acc += (output.argmax(1) == labels).float().mean().item()
            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * running_acc / (n_batch+1)})
            pbar.update() # Update progress bar
    return running_loss / len(trainloader), running_acc / len(trainloader) # return loss and accuracy for this epoch

# Define function to call for each validation epoch (one complete pass over the validation set)
def validate(model, valloader, criterion, device, tag='Val'):
    model.eval() # set model to evaluation mode (e.g. turn off dropout, batchnorm, etc.)
    running_loss = 0; running_acc = 0
    with torch.no_grad(): # no need to compute gradients for validation
        with tqdm(total=len(valloader), desc=tag, unit="batch") as pbar:
            for n_batch, (images, labels) in enumerate(valloader): # Iterate over batches
                images, labels = images.to(device), labels.to(device) # Move batch to device
                output = model(images) # Forward pass
                loss = criterion(output, labels) # Compute loss
                running_loss += loss.item() 
                running_acc += (output.argmax(1) == labels).float().mean().item()
                pbar.set_postfix({'loss': loss.item(), 'acc': 100. * running_acc / (n_batch+1)})
                pbar.update() # Update progress bar
    return running_loss / len(valloader), running_acc / len(valloader)  # return loss and accuracy for this epoch

# Run training and validation loop
# Save the best model based on validation accuracy
n_epochs = 20
best_acc = -1
train_loss_history = []; train_acc_history = []
val_loss_history = []; val_acc_history = []
for epoch in range(n_epochs): # Iterate over epochs
    print(f"\nEpoch {epoch+1} of {n_epochs}")
    if epoch == n_epochs // 2:
        lr = optimizer.param_groups[0]['lr']
        print(f'Reducing learning rate from {lr} to {lr/4}')
        optimizer.param_groups[0]['lr'] /= 4
    train_loss, train_acc  = train(model, trainloader, criterion, optimizer, device) # Train
    train_loss, train_acc  = validate(model, train_eval_loader, criterion, device, tag='Train Eval') # Evaluate on Train data
    val_loss, val_acc = validate(model, valloader, criterion, device) # Validate
    train_loss_history.append(train_loss); train_acc_history.append(train_acc)
    val_loss_history.append(val_loss); val_acc_history.append(val_acc)
    if val_acc > best_acc: # Save best model
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt") # saving model parameters ("state_dict") saves memory and is faster than saving the entire model

epochs = torch.arange(n_epochs)

# plot training and validation loss
plt.figure()
plt.plot(epochs, train_loss_history, label='train_loss')
plt.plot(epochs, val_loss_history, label='val_loss')
plt.xlabel('epochs')
plt.ylabel('Multiclass Cross Entropy Loss')
plt.title(f'Loss with miniVGG model')
plt.legend()
plt.show()

# plot training and validation accuracy
plt.figure()
plt.plot(epochs, train_acc_history, label='train_acc')
plt.plot(epochs, val_acc_history, label='val_acc')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.title(f'Accuracy with miniVGG model; Regularizer: {reg_val : 3.2g}')
plt.legend()
plt.show()

# Load the best model and evaluate on test set
model.load_state_dict(torch.load("best_model.pt"))
test_loss, test_acc = validate(model, testloader, criterion, device)
print(f"Test accuracy: {test_acc:.4f}")

model.eval() # set model to evaluation mode 
img = np.random.randint(10000)
with torch.no_grad():
    image, label = testset[img] # get first image and label from test set
    image = image.unsqueeze(1)  # add the batch dimension
    image = image.to(device)  # move image to device
    output = model(image) # forward pass
pred = output.argmax(1) # get predicted class
print(f"Test Image Number: {img}, Predicted class: {testset.classes[pred.item()]}")
# plot image 
plt.figure()
plt.imshow(image.cpu().numpy().squeeze(), cmap='gray')
plt.show()


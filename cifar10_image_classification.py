from coutils import fix_random_seed
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

# for plotting
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

NUM_TRAIN = 49000
transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

cifar10_train = dset.CIFAR10('./datasets', train=True, download=True,
                             transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

cifar10_val = dset.CIFAR10('./datasets', train=True, download=True,
                           transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=64, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

cifar10_test = dset.CIFAR10('./datasets', train=False, download=True, 
                            transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)

dtype = torch.float
ltype = torch.long

if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 100

print('using device:', device)

def flatten(x, start_dim=1, end_dim=-1):
    return x.flatten(start_dim=start_dim, end_dim=end_dim)

def two_layer_fc(x, params):
    # first we flatten the image
    x = flatten(x)  # shape: [batch_size, C x H x W]
    w1, b1, w2, b2 = params
    x = F.relu(F.linear(x, w1.T, b1))
    x = F.linear(x, w2.T, b2)
    return x

def two_layer_fc_test():
    hidden_layer_size = 42
    x = torch.zeros((64, 3, 16, 16), dtype=dtype)  # minibatch size 64, feature dimension 3*16*16
    w1 = torch.zeros((3*16*16, hidden_layer_size), dtype=dtype)
    b1 = torch.zeros((hidden_layer_size,), dtype=dtype)
    w2 = torch.zeros((hidden_layer_size, 10), dtype=dtype)
    b2 = torch.zeros((10,), dtype=dtype)
    scores = two_layer_fc(x, [w1, b1, w2, b2])
    print('Output size:', list(scores.size()))  # you should see [64, 10]

two_layer_fc_test()

def three_layer_convnet(x, params):
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    scores = None
    x = F.relu(F.conv2d(x, conv_w1, conv_b1, padding=2))
    x = F.relu(F.conv2d(x, conv_w2, conv_b2, padding=1))
    x = flatten(x)
    scores = F.linear(x, fc_w.T, fc_b)
    return scores

def three_layer_convnet_test():
    x = torch.zeros((64, 3, 32, 32), dtype=dtype)  # minibatch size 64, image size [3, 32, 32]

    conv_w1 = torch.zeros((6, 3, 5, 5), dtype=dtype)  # [out_channel, in_channel, kernel_H, kernel_W]
    conv_b1 = torch.zeros((6,))  # out_channel
    conv_w2 = torch.zeros((9, 6, 3, 3), dtype=dtype)  # [out_channel, in_channel, kernel_H, kernel_W]
    conv_b2 = torch.zeros((9,))  # out_channel

    # you must calculate the shape of the tensor after two conv layers, before the fully-connected layer
    fc_w = torch.zeros((9 * 32 * 32, 10))
    fc_b = torch.zeros(10)

    scores = three_layer_convnet(x, [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b])
    print(scores.size())  # you should see [64, 10]
    return scores.size()
three_layer_conv_size = three_layer_convnet_test()

def random_weight(shape):
    if len(shape) == 2:  # FC weight
        fan_in = shape[0]
    else:
        fan_in = np.prod(shape[1:]) # conv weight [out_channel, in_channel, kH, kW]
    # randn is standard normal distribution generator. 
    w = torch.randn(shape, device=device, dtype=dtype) * np.sqrt(2. / fan_in)
    w.requires_grad = True
    return w

def zero_weight(shape):
    return torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)

print(random_weight((3, 5)))

def check_accuracy_part2(loader, model_fn, params):
    split = 'val' if loader.dataset.train else 'test'
    print('Checking accuracy on the %s set' % split)
    num_correct, num_samples = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=ltype)
            scores = model_fn(x, params)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
        return acc

def train_part2(model_fn, params, learning_rate):
    acc = 0
    for t, (x, y) in enumerate(loader_train):
        # Move the data to the proper device (GPU or CPU)
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=ltype)

        # Forward pass: compute scores and loss
        scores = model_fn(x, params)
        loss = F.cross_entropy(scores, y)

        loss.backward()

        with torch.no_grad():
            for w in params:
                if w.requires_grad:
                    w -= learning_rate * w.grad
                    w.grad.zero_() # Manually zero the gradients after running the backward pass

        if t % print_every == 0 or t == len(loader_train)-1:
            print('Iteration %d, loss = %.4f' % (t, loss.item()))
            acc = check_accuracy_part2(loader_val, model_fn, params)
            print()
    return acc

fix_random_seed(0)

C, H, W = 3, 32, 32
num_classes = 10

hidden_layer_size = 4000
learning_rate = 1e-2

w1 = random_weight((C*H*W, hidden_layer_size))
b1 = zero_weight(hidden_layer_size)
w2 = random_weight((hidden_layer_size, num_classes))
b2 = zero_weight(num_classes)

train_part2(two_layer_fc, [w1, b1, w2, b2], learning_rate)

channel_1 = 32
channel_2 = 16
kernel_size_1 = 5
kernel_size_2 = 3

learning_rate = 3e-3

conv_w1 = random_weight((channel_1, C, kernel_size_1, kernel_size_1))
conv_b1 = zero_weight(channel_1)
conv_w2 = random_weight((channel_2, channel_1, kernel_size_2, kernel_size_2))
conv_b2 = zero_weight(channel_2)
fc_w = random_weight((channel_2 * H * W, num_classes))
fc_b = zero_weight(num_classes)

params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
acc_part2 = train_part2(three_layer_convnet, params, learning_rate)

class Linear(nn.Module):
    def __init__(self, dim_in, dim_out, device):
        super(Linear, self).__init__()
        self.weights = nn.Parameter(torch.randn([dim_out, dim_in], dtype=torch.float32, device=device)*0.1, requires_grad=True)
        self.bias = nn.Parameter(torch.zeros([dim_out], dtype=torch.float32, device=device), requires_grad=True)
        self.dim_out = dim_out
        self.device = device
    
    def linear_forward(self, X):
        Y = torch.mm(X, self.weights.T) + self.bias
        return Y
    
    def forward(self, X):
        return self.linear_forward(X)

# correctness checking
x = torch.arange(50).view(5, 10).float()
my_linear = Linear(10, 3, torch.device('cpu'))
y = my_linear(x)
print(y)

class Conv2D(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, device):
        super(Conv2D, self).__init__()
        # initialize kernel and bias
        self.kernel = nn.Parameter(torch.randn([dim_out, dim_in]+kernel_size, dtype=torch.float32, device=device)*0.1, requires_grad=True)
        self.bias = nn.Parameter(torch.zeros([dim_out], dtype=torch.float32, device=device), requires_grad=True)
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device
    
    def conv2d_forward(self, X):
        N, _, H_in, W_in = X.shape
        KH, KW = self.kernel_size
        X_padded = torch.nn.functional.pad(X, self.padding)

        H_out = (H_in + self.padding[0] + self.padding[1] - KH) // self.stride[0] + 1
        W_out = (W_in + self.padding[2] + self.padding[3] - KW) // self.stride[1] + 1
        Y = torch.zeros((N, self.dim_out, H_out, W_out), device=self.device)

        for i in range(H_out):
            for j in range(W_out):
                X_slice = X_padded[:, :, i*self.stride[0]:i*self.stride[0]+KH, j*self.stride[1]:j*self.stride[1]+KW]
                Y[:, :, i, j] = torch.sum(X_slice.unsqueeze(1) * self.kernel.unsqueeze(0), dim=(2, 3, 4))
        Y += self.bias.view(1, -1, 1, 1)
        return Y

    def forward(self, x):
        return self.conv2d_forward(x)

# correctness checking
x = torch.arange(50).view(2,1,5,5).float()
my_conv = Conv2D(1,2,[3,3],[3,3],[1,1,1,1], torch.device('cpu'))
y = my_conv(x)
print(y)

class TwoLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # assign layer objects to class attributes
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_ 
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
  
    def forward(self, x):
        x = flatten(x)
        scores = self.fc2(F.relu(self.fc1(x)))
        return scores

def test_TwoLayerFC():
    input_size = 3*16*16
    x = torch.zeros((64, input_size), dtype=dtype)  # minibatch size 64, feature dimension 3*16*16
    model = TwoLayerFC(input_size, 42, 10)
    scores = model(x)
    print('Architecture:')
    print(model)
    print('Output size:', list(scores.size()))  # you should see [64, 10]
test_TwoLayerFC()

class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, channel_1, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1)
        self.fc = nn.Linear(channel_2 * 32 * 32, num_classes)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = flatten(x)
        scores = self.fc(x)
        return scores

def test_ThreeLayerConvNet():
    x = torch.zeros((64, 3, 32, 32), dtype=dtype)  # minibatch size 64, image size [3, 32, 32]
    model = ThreeLayerConvNet(in_channel=3, channel_1=12, channel_2=8, num_classes=10)
    scores = model(x)
    print(model)
    print('Output size:', list(scores.size()))  # you should see [64, 10]
test_ThreeLayerConvNet()

def check_accuracy_part34(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=ltype)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc

def train_part34(model, optimizer, epochs=1):
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    acc = 0
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                acc = check_accuracy_part34(loader_val, model)
                print()
    return acc

def adjust_learning_rate(optimizer, lrd, epoch, schedule):
    if epoch in schedule:
        for param_group in optimizer.param_groups:
            print('lr decay from {} to {}'.format(param_group['lr'], param_group['lr'] * lrd))
            param_group['lr'] *= lrd

def train_part345(model, optimizer, epochs=1, learning_rate_decay=.1, schedule=[], verbose=True):
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    num_iters = epochs * len(loader_train)
    if verbose:
        num_prints = num_iters // print_every + 1
    else:
        num_prints = epochs
    acc_history = torch.zeros(num_prints, dtype=torch.float)
    iter_history = torch.zeros(num_prints, dtype=torch.long)
    for e in range(epochs):
    
        adjust_learning_rate(optimizer, learning_rate_decay, e, schedule)
    
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=ltype)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tt = t + e * len(loader_train)

            if verbose and (tt % print_every == 0 or (e == epochs-1 and t == len(loader_train)-1)):
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, tt, loss.item()))
                acc = check_accuracy_part34(loader_val, model)
                acc_history[tt // print_every] = acc
                iter_history[tt // print_every] = tt
                print()
            elif not verbose and (t == len(loader_train)-1):
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, tt, loss.item()))
                acc = check_accuracy_part34(loader_val, model)
                acc_history[e] = acc
                iter_history[e] = tt
                print()
    return acc_history, iter_history

hidden_layer_size = 4000
learning_rate = 1e-2
weight_decay = 1e-4

model = TwoLayerFC(C*H*W, hidden_layer_size, num_classes)
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      weight_decay=weight_decay)
_ = train_part345(model, optimizer)

C = 3
channel_1 = 32
channel_2 = 16

learning_rate = 3e-3
weight_decay = 1e-4

model = ThreeLayerConvNet(C, channel_1, channel_2, num_classes)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
three_layer_conv_acc_history, _ = train_part345(model, optimizer)

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

fix_random_seed(0)

C, H, W = 3, 32, 32
num_classes = 10

hidden_layer_size = 4000
learning_rate = 1e-2
momentum = 0.5

model = nn.Sequential(OrderedDict([
    ('flatten', Flatten()),
    ('fc1', nn.Linear(C*H*W, hidden_layer_size)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(hidden_layer_size, num_classes)),
]))

print('Architecture:')
print(model)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                      weight_decay=weight_decay,
                      momentum=momentum, nesterov=True)
_ = train_part345(model, optimizer)

kernel_size_1 = 5
pad_size_1 = 2
kernel_size_2 = 3
pad_size_2 = 1

model = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(C, channel_1, kernel_size_1, padding=pad_size_1)),
    ('relu1', nn.ReLU()),
    ('conv2', nn.Conv2d(channel_1, channel_2, kernel_size_2, padding=pad_size_2)),
    ('relu2', nn.ReLU()),
    ('flatten', Flatten()),
    ('fc', nn.Linear(channel_2 * H * W, num_classes))
]))
optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                      weight_decay=weight_decay, 
                      momentum=momentum, nesterov=True)

print('Architecture:')
print(model)
three_layer_conv_seq_acc_history, _ = train_part345(model, optimizer)

channel_1 = 32
channel_2 = 16
kernel_size = 3
pad_size = 1
learning_rate = 1e-2
weight_decay = 1e-4
momentum = 0.5

best_model = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(C, channel_1, kernel_size_1, padding=pad_size_1)),
    ('bn1', nn.BatchNorm2d(channel_1)),
    ('relu1', nn.ReLU()),
    ('pool1', nn.MaxPool2d(2)),
    ('conv2', nn.Conv2d(channel_1, channel_2, kernel_size_2, padding=pad_size_2)),
    ('bn2', nn.BatchNorm2d(channel_2)),
    ('relu2', nn.ReLU()),
    ('pool2', nn.MaxPool2d(2)),
    ('dropout', nn.Dropout(p=0.5)),
    ('flatten', Flatten()),
    ('fc1', nn.Linear(channel_2 * (H // 4) * (W // 4), channel_2 * (H // 4) * (W // 4) // 2)),
    ('relu3', nn.ReLU()), 
    ('fc2', nn.Linear(channel_2 * (H // 4) * (W // 4) // 2, num_classes)),
]))
optimizer = optim.SGD(best_model.parameters(), lr=learning_rate, 
                      weight_decay=weight_decay, 
                      momentum=momentum, nesterov=True)

train_part34(best_model, optimizer, epochs=10)
acc_final = check_accuracy_part34(loader_test, best_model)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from googlenet import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', '-e', default=1, type=int, help='epoch numbers')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(root='./data',
                                         train=True,
                                         download=True,
                                         transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=128,
                                          shuffle=True,
                                          num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data',
                                        train=False,
                                        download=True,
                                        transform=transform_test)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=100,
                                         shuffle=False,
                                         num_workers=2)

# Model
print('==> Building model..')
net1 = GoogLeNet(naive=True)
net1 = net1.to(device)
net2 = GoogLeNet(naive=False)
net2 = net2.to(device)
if device == 'cuda':
    net1 = torch.nn.DataParallel(net1)
    net2 = torch.nn.DataParallel(net2)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer1 = optim.SGD(net1.parameters(),
                       lr=args.lr,
                       momentum=0.9,
                       weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(),
                       lr=args.lr,
                       momentum=0.9,
                       weight_decay=5e-4)


# Training
def train(net, optimizer, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct_5 = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted_5 = torch.topk(outputs, 5)
        total += targets.size(0)

        predicted_5 = predicted_5.t()
        tmp = predicted_5.eq(targets.view(1, -1).expand_as(predicted_5))
        correct_5 += tmp[:5].view(-1).float().sum(0).item()

        progress_bar(
            batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (train_loss /
             (batch_idx + 1), 100. * correct_5 / total, correct_5, total))
    return correct_5, total


def test(net, epoch):
    net.eval()
    test_loss = 0
    correct_5 = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted_5 = torch.topk(outputs, 5)
            total += targets.size(0)

            predicted_5 = predicted_5.t()
            tmp = predicted_5.eq(targets.view(1, -1).expand_as(predicted_5))
            correct_5 += tmp[:5].view(-1).float().sum(0).item()

            progress_bar(
                batch_idx, len(testloader),
                'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (test_loss /
                 (batch_idx + 1), 100. * correct_5 / total, correct_5, total))
    return correct_5, total


total_train1 = total_test1 = total_train2 = total_test2 = 0
correct_train1 = correct_test1 = correct_train2 = correct_test2 = 0

print('\ngoogLeNet_Naive', end='')
for epoch in range(0, args.epoch):
    c_train, n_train = train(net1, optimizer1, epoch)
    total_train1 += n_train
    correct_train1 += c_train

    c_test, n_test = test(net1, epoch)
    total_test1 += n_test
    correct_test1 += c_test

print('\ngoogLeNet_BottleLayer', end='')
for epoch in range(0, args.epoch):
    c_train, n_train = train(net2, optimizer2, epoch)
    total_train2 += n_train
    correct_train2 += c_train

    c_test, n_test = test(net2, epoch)
    total_test2 += n_test
    correct_test2 += c_test

trainalbe_params1 = sum(p.numel() for p in net1.parameters()
                        if p.requires_grad)
trainalbe_params2 = sum(p.numel() for p in net2.parameters()
                        if p.requires_grad)

print('\nNet\t\t\tTop-5 error (train)\tTop-5 error (test)\t# of param')
print('googLeNet_Naive\t\t%-.3f%%\t\t\t%-.3f%%\t\t\t%d' %
      (100 * (1 - (correct_train1 / total_train1)),
       (100 * (1 - (correct_test1 / total_test1))), trainalbe_params1))
print('googLeNet_BottleLayer\t%-.3f%%\t\t\t%-.3f%%\t\t\t%d' %
      (100 * (1 - (correct_train2 / total_train2)),
       (100 * (1 - (correct_test2 / total_test2))), trainalbe_params2))

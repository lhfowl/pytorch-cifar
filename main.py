'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from subset_class import sub_dataset

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--num_train', default=500 )
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
#trainloader = torch.utils.data.DataLoader(
#    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

num_runs = 30
accs = np.zeros(14)
net_names = []
for i in range(num_runs):
    new_trainset = sub_dataset(trainset, 500)
    trainloader = torch.utils.data.DataLoader(
        new_trainset, batch_size=32, shuffle=True, num_workers=2)
    for j in range(14):
        # Model
        print('==> Building model..')
        if j==0:
            net = VGG('VGG19')
            net_names.append('VGG19')
        elif j==1:
            net = ResNet18()
            net_names.append('ResNet18')
        elif j==2:
            #net = PreActResNet18()
            net_names.append('PreActResNet18')
            continue
        elif j==3:
            net = GoogLeNet()
            net_names.append('GoogLeNet')
        elif j==4:
            net = DenseNet121()
            net_names.append('DenseNet121')
        elif j==5:
            net = ResNeXt29_2x64d()
            net_names.append('ResNeXt29')
        elif j==6:
            net = MobileNet()
            net_names.append('MobileNet')
        elif j==7:
            net = MobileNetV2()
            net_names.append('MobileNetV2')
        elif j==8:
            net = DPN92()
            net_names.append('DPN92')
        elif j==9:
            #net = ShuffleNetG2()
            net_names.append('ShuffleNetG2')
            continue
        elif j==10:
            net = SENet18()
            net_names.append('SENet18')
        elif j==11:
            net = ShuffleNetV2(1)
            net_names.append('ShuffleNetV2')
        elif j==12:
            net = EfficientNetB0()
            net_names.append('EfficientNetB0')
        elif j==13:
            net = RegNetX_200MF()
            net_names.append('RegNetX_200')
        net = net.to(device)

        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        if args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.pth')
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)


        # Training
        def train(epoch):
            print('\nEpoch: %d' % epoch)
            net.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


        def test(epoch):
            global best_acc
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                 % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            acc = 100.*correct/total
            return acc

        for epoch in range(start_epoch, start_epoch+100):
            train(epoch)
            #test(epoch)
        acc = test(epoch)
        accs[j] += acc

accs/= num_runs
print(accs)

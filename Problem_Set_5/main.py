import torch
import torch.nn as nn
import argparse
import torchvision
import torchvision.transforms as transforms
import os
PATH = './model.pt'
class LinearClassifier(nn.Module):
    # define a linear classifier
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # inchannels: dimenshion of input data. For example, a RGB image [3x32x32] is converted to vector [3 * 32 * 32], so dimenshion=3072
        # out_channels: number of categories. For CIFAR-10, it's 10
        self.linear = nn.Linear(in_channels, out_channels)
    def forward(self, x: torch.Tensor):
        return self.linear(x)


class FCNN(nn.Module):
    # def a full-connected neural network classifier
    def __init__(self, in_channels, hidden_channels, out_channels) -> None:
        super().__init__()
        # inchannels: dimenshion of input data. For example, a RGB image [3x32x32] is converted to vector [3 * 32 * 32], so dimenshion=3072
        # hidden_channels
        # out_channels: number of categories. For CIFAR-10, it's 10

        # full connected layer
        # activation function
        # full connected layer
        # ......
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor): 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

def svmloss(scores: torch.Tensor, label: torch.Tensor):
    '''
    compute SVM loss
    input:
        scores: output of model 
        label: true label of data
    return:
        svm loss
    '''
    correct_scores = scores[torch.arange(scores.size(0)), label].view(-1, 1)
    margins = torch.clamp(scores - correct_scores + 1, min=0)
    margins[torch.arange(scores.size(0)), label] = 0  # Ignore the correct class
    loss = margins.sum() / scores.size(0)

    return loss

def crossentropyloss(logits: torch.Tensor, label: torch.Tensor):
    '''
    Object: implement Cross Entropy loss function
    input:
        logits: output of model, (unnormalized log-probabilities). shape: [batch_size, c]
        label: true label of data. shape: [batch_size]
    return: 
        cross entropy loss
    '''

    exp_logits = torch.exp(logits)
    softmax_probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)
    loss = -torch.log(softmax_probs[torch.arange(logits.size(0)), label]).mean()
    return loss


def train(model, loss_function, optimizer, scheduler, args):
    '''
    Model training function
    input: 
        model: linear classifier or full-connected neural network classifier
        loss_function: SVM loss of Cross-entropy loss
        optimizer: Adamw or SGD
        scheduler: step or cosine
        args: configuration
    '''
    if os.path.exists(PATH):
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # create dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # create dataloader
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    # for-loop 
    epoch_cnt = 2
    running_loss = 0.0
    for epoch in range(epoch_cnt):
        # train
        # get the inputs; data is a list of [inputs, labels]
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = nn.Flatten()(inputs)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(input) 
            # loss backward
            loss = loss_function(outputs, labels)
            loss.backward()
            # optimize
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        # adjust learning rate
        scheduler.step()
        # test
        correct = 0
        total = 0
        with torch.no_grad():
            # forward
            for data in testloader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # save checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, PATH)

def test(model, loss_function, args):
    '''
    input: 
        model: linear classifier or full-connected neural network classifier
        loss_function: SVM loss of Cross-entropy loss
    '''
    # load checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    # create testing dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 4
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # create dataloader
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    # test
    correct = 0
    total = 0
    with torch.no_grad():
        # forward
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='The configs')

    parser.add_argument('--run', type=str, default='train')
    parser.add_argument('--model', type=str, default='linear')
    parser.add_argument('--loss', type=str, default='svmloss')
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='step')
    args = parser.parse_args()

    # create model
    if args.model == 'linear':
        model = LinearClassifier(3 * 32 * 32, 10)
    elif args.model == 'fcnn':
        model = FCNN(3 * 32 * 32, 1024, 10)
    else: 
        raise AssertionError

    # create optimizer
    if args.optimizer == 'adamw':
        # create Adamw optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
    elif args.optimizer == 'sgd':
        # create SGD optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
    else:
        raise AssertionError
    
    # create scheduler
    if args.scheduler == 'step':
        # create torch.optim.lr_scheduler.StepLR scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    elif args.scheduler == 'cosine':
        # create torch.optim.lr_scheduler.CosineAnnealingLR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, step_size=100, gamma=0.1)
    else:
        raise AssertionError

    if args.run == 'train':
        train(model, eval(args.loss), optimizer, scheduler, args)
    elif args.run == 'test':
        test(model, eval(args.loss), args)
    else: 
        raise AssertionError
    
# You need to implement training and testing function that can choose model, optimizer, scheduler and so on by command, such as:
# python main.py --run=train --model=fcnn --loss=crossentropyloss --optimizer=adamw --scheduler=step



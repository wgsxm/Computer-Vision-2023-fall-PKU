import torch
import torch.nn as nn
import argparse


class LinearClassifier(nn.Module):
    # define a linear classifier
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # inchannels: dimenshion of input data. For example, a RGB image [3x32x32] is converted to vector [3 * 32 * 32], so dimenshion=3072
        # out_channels: number of categories. For CIFAR-10, it's 10

    def forward(self, x: torch.Tensor):
        return 


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

    def forward(self, x: torch.Tensor): 
        return 
    

def svmloss(scores: torch.Tensor, label: torch.Tensor):
    '''
    compute SVM loss
    input:
        scores: output of model 
        label: true label of data
    return:
        svm loss
    '''

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
    # softmax function
    # cross entropy

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
    # create dataset

    # create dataloader

    # for-loop 
        # train
            # get the inputs; data is a list of [inputs, labels]

            # zero the parameter gradients

            # forward

            # loss backward

            # optimize

        # adjust learning rate

        # test
            # forward

    # save checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)

def test(model, loss_function, args):
    '''
    input: 
        model: linear classifier or full-connected neural network classifier
        loss_function: SVM loss of Cross-entropy loss
    '''
    # load checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
    # create testing dataset
    # create dataloader
    # test
        # forward

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
        model = 
    elif args.model == 'fcnn':
        model = 
    else: 
        raise AssertionError

    # create optimizer
    if args.optimizer == 'adamw':
        # create Adamw optimizer
        optimizer = 
    elif args.optimizer == 'sgd':
        # create SGD optimizer
        optimizer = 
    else:
        raise AssertionError
    
    # create scheduler
    if args.scheduler == 'step':
        # create torch.optim.lr_scheduler.StepLR scheduler
        scheduler = 
    elif args.scheduler == 'cosine':
        # create torch.optim.lr_scheduler.CosineAnnealingLR scheduler
        scheduler = 
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



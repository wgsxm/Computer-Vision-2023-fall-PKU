import torch
import torch.nn.functional as F
from models import VGG, ResNet, ResNext
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train(model, args):
    '''
    Model training function
    input: 
        model: linear classifier or full-connected neural network classifier
        args: configuration
    '''
    # create dataset, data augmentation

    # create dataloader

    # create optimizer

    # create scheduler 

    # ctreat summary writer

        # train

            # get the inputs; data is a list of [inputs, labels]

            # zero the parameter gradients

            # forward

            # backward

            # optimize

        # scheduler adjusts learning rate

        # test

    # save checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)


    def test(model, args):
        '''
        input: 
            model: linear classifier or full-connected neural network classifier
            args: configuration
        '''
        # load checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
        # create testing dataset
        # create dataloader
        # test
            # forward

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='The configs')
    parser.add_argument()
    args = parser.parse_args()

    model = 
    # train / test

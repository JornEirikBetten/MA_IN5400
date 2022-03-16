
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
#import matplotlib.pyplot as plt

from torch import Tensor

import time
import os
import numpy as np

import PIL.Image
import sklearn.metrics
from sklearn import preprocessing

from typing import Callable, Optional
#from RainforestDataset import RainforestDataset, ChannelSelect
#rom YourNetwork import SingleNetwork
from torchvision.models import resnet18
import csv


def get_classes_list():
    classes = ['clear', 'cloudy', 'haze', 'partly_cloudy',
               'agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
               'blow_down', 'conventional_mine', 'cultivation', 'habitation',
               'primary', 'road', 'selective_logging', 'slash_burn', 'water']
    return classes, len(classes)


class TwoNetworks(nn.Module):
    '''
    This class takes two pretrained networks,
    concatenates the high-level features before feeding these into
    a linear layer.

    functions: forward
    '''
    def __init__(self, pretrained_net1, pretrained_net2):
        super(TwoNetworks, self).__init__()

        _, num_classes = get_classes_list()

        # TODO select all parts of the two pretrained networks, except for
        # the last linear layer.
        self.fully_conv1 = nn.Sequential(*(list(pretrained_net1.children())[:-1]))
        self.fully_conv2 = nn.Sequential(*(list(pretrained_net2.children())[:-1]))
        self.fully_conv2[0] = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)



        # TODO create a linear layer that has in_channels equal to
        # the number of in_features from both networks summed together.
        print("pretrained_net1.fc.in_feat: ", pretrained_net1.fc.in_features)
        print("pretrained_net2.fc.in_feat: ", pretrained_net2.fc.in_features)

        self.linear = nn.Linear(pretrained_net1.fc.in_features + pretrained_net2.fc.in_features, num_classes)


    def forward(self, inputs1, inputs2):
        # TODO feed the inputs through the fully convolutional parts
        # of the two networks that you initialised above, and then
        # concatenate the features before the linear layer.
        # And return the result.
        net1 = self.fully_conv1(inputs1)
        net2 = self.fully_conv2(inputs2)
        print("Shape net 1: ", net1.shape)
        print("Shape net 2: ", net2.shape)

        net = torch.cat((net1, net2), axis=1)
        output = self.linear(net)

        return output

class SingleNetwork(nn.Module):
    '''
    This class takes one pretrained network,
    the first conv layer can be modified to take an extra channel.

    functions: forward
    '''

    def __init__(self, pretrained_net, weight_init=None):
        super(SingleNetwork, self).__init__()

        _, num_classes = get_classes_list()

        if weight_init is not None:
            current_weights = pretrained_net.conv1.weight
            if weight_init == 'kaiming-he':
                weights_fourth_layer = torch.empty(64, 1, 7, 7)
            else:
                weights_fourth_layer = torch.randn(64, 1, 7, 7)
            new_weights = torch.cat((current_weights, weights_fourth_layer), dim=1)

            new_conv1 = nn.Conv2d(4, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
            new_conv1.weight = torch.nn.Parameter(new_weights)
            pretrained_net.conv1 = new_conv1

        pretrained_net.fc = nn.Linear(pretrained_net.fc.in_features, num_classes)
        self.net = pretrained_net

        """
        if weight_init is not None:
            # TODO Here we want an additional channel in the weights tensor, specifically in the first
            # conv2d layer so that there are weights for the infrared channel in the input aswell.
            current_weights = pretrained_net.conv1.weight
            new_weights = torch.cat((current_weights, torch.randn(64, 1, 7, 7)

            if weight_init == "kaiminghe":
              pass


            # TODO Create a new conv2d layer, and set the weights to be
            # what you created above. You will need to pass the weights to
            # torch.nn.Parameter() so that the weights are considered
            # a model parameter.
            # eg. first_conv_layer.weight = torch.nn.Parameter(your_new_weights)
            new_conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            new_conv1.weight = torch.nn.Parameter(new_weights)
            pretrained_net.conv1 = new_conv1

        # TODO Overwrite the last linear layer.
        pretrained_net.fc = nn.Linear(pretrained_net.fc.in_features, num_classes)

        self.net = pretrained_net
        """

    def forward(self, inputs):
        return self.net(inputs)

if __name__=='__main__':
    model1 = resnet18(pretrained=True)
    model2 = resnet18(pretrained=True)
    model = TwoNetworks(model1, model2)

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import PIL.Image
import pandas as pd
import numpy as np
from sklearn import preprocessing
from torchvision.models import resnet18


def get_classes_list():
    classes = ['clear', 'cloudy', 'haze', 'partly_cloudy',
               'agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
               'blow_down', 'conventional_mine', 'cultivation', 'habitation',
               'primary', 'road', 'selective_logging', 'slash_burn', 'water']
    return classes, len(classes)

root_dir = "/home/jeb/Documents/IN5400/student_version/"

csv_file = root_dir + "rainforest/train_v2.csv"

dlabels = pd.read_csv(csv_file)

print(dlabels["tags"][0])



classes, num_classes = get_classes_list()

N = 10

labels = []

for i in range(N):
	tags = dlabels["tags"][i]
	tags = tags.split(' ')
	label = []
	for j in range(num_classes):
		if classes[j] in tags:
			label.append(1)
		else:
			label.append(0)
	labels.append(label)

print(labels)


labels_bin = preprocessing.binarize(labels)
print("First row: ", labels_bin[0])
print(labels_bin)


model = resnet18(pretrained=True)
print("Full model: ", model)
print(model.conv1)
print(model.conv1.weight.shape)
print(model.conv1.bias)

pred = torch.ones((2, 3))
pred[0, 1] = 0
pred[0, 0] = 0.5
targ = torch.ones((2, 3))*2
targ[0,0] = 50
print(pred)
print(targ)
print(torch.multiply(pred, targ))

print(np.empty((0, 17)).shape)

target = [[0, 0, 5, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]]
target = preprocessing.binarize(target)
print(target)

def loss(output, target):
    #TODO
    off_target = 1-target
    log_pred = torch.log(input_)
    log_pred_1 = torch.log(input_-1)
    cross_entropy_all = torch.multiply(target, log_pred) + torch.multiply(off_target, log_pred_1)
    loss = -torch.sum(cross_entropy_all)
    loss = loss/(input_.shape[0])
    return loss


print(os.getcwd())
print(__file__)

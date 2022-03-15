

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
from RainforestDataset import RainforestDataset, ChannelSelect
from YourNetwork import SingleNetwork
from torchvision.models import resnet18
import csv


activation = nn.Sigmoid()
torch.set_default_dtype(torch.float64)


def train_epoch(model, trainloader, criterion, device, optimizer):

    #TODO model.train() or model.eval()?
    model.train(True)
    numcl = 17
    concat_pred = np.empty((0, numcl))
    concat_labels = np.empty((0, numcl))
    avgprecs=np.zeros(numcl)
    losses = []
    for batch_idx, data in enumerate(trainloader):
        if (batch_idx %100==0) and (batch_idx>=100):
          print('at batchidx',batch_idx)

        # TODO calculate the loss from your minibatch.
        # If you are using the TwoNetworks class you will need to copy the infrared
        # channel before feeding it into your model.
        optimizer.zero_grad()
        inputs = data['image'].to(device)
        target = data['label'].to(device)
        target = target.double()
        prediction = model(inputs)
        prediction = activation(prediction)
        prediction = prediction.double()
        loss = criterion(prediction, target)
        #print(loss)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        cpuout = Tensor.cpu(prediction).detach().numpy()
        labels = Tensor.cpu(target)
        cpuout=np.nan_to_num(cpuout)
        preds = preprocessing.binarize(cpuout, threshold=0.5, copy=True)
        concat_pred = np.concatenate((concat_pred, cpuout), axis=0)
        concat_labels = np.concatenate((concat_labels, labels), axis=0)
    for c in range(numcl):
      avgprecs[c]= sklearn.metrics.average_precision_score(concat_labels[:, c], concat_pred[:, c])# TODO, nope it is not sklearn.metrics.precision_score

    return np.mean(losses), np.nan_to_num(avgprecs)


def evaluate_meanavgprecision(model, dataloader, criterion, device, numcl):

    #TODO model.train() or model.eval()?
    model.eval()
    curcount = 0
    accuracy = 0



    concat_pred = np.empty((0, numcl)) #prediction scores for each class. each numpy array is a list of scores. one score per image
    concat_labels = np.empty((0, numcl)) #labels scores for each class. each numpy array is a list of labels. one label per image
    avgprecs=np.zeros(numcl) #average precision for each class
    fnames = [] #filenames as they come out of the dataloader

    with torch.no_grad():
      losses = []
      for batch_idx, data in enumerate(dataloader):
          if (batch_idx%100==0) and (batch_idx>=100):
            print('at val batchindex: ', batch_idx)

          inputs = data['image'].to(device).double()
          outputs = model(inputs)
          outputs = activation(outputs)
          labels = data['label'].double()
          loss = criterion(outputs, labels.to(device))
          losses.append(loss.item())

          # This was an accuracy computation
          cpuout= outputs.to('cpu')
          cpuout=np.nan_to_num(cpuout)
          preds = preprocessing.binarize(cpuout, threshold=0.5, copy=True)
          corrects = np.sum([preds[i,j] == labels[i,j] for i in range(preds.shape[0]) for j in range(preds.shape[1])])
          accuracy = accuracy*( curcount/ float(curcount+labels.shape[0]) ) + float(corrects)* ( curcount/ float(curcount+labels.shape[0]) )
          curcount += labels.shape[0]

          # TODO: collect scores, labels, filenames
          concat_pred = np.concatenate((concat_pred, cpuout), axis=0)
          concat_labels = np.concatenate((concat_labels, labels), axis=0)
          for fname in data['filename']:
              fnames.append(fname)

    for c in range(numcl):
      avgprecs[c]= sklearn.metrics.average_precision_score(concat_labels[:, c], concat_pred[:, c])# TODO, nope it is not sklearn.metrics.precision_score


    return np.nan_to_num(avgprecs), np.mean(losses), concat_labels, concat_pred, fnames


def traineval2_model_nocv(dataloader_train, dataloader_test ,  model ,  criterion, optimizer, scheduler, num_epochs, device, numcl):

  best_measure = 0
  best_epoch =-1
  trainlosses=[]
  trainperfs=[]
  testlosses=[]
  testperfs=[]

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)


    avgloss, train_mAP = train_epoch(model,  dataloader_train,  criterion,  device , optimizer )
    trainlosses.append(avgloss)
    trainperfs.append(train_mAP)


    if scheduler is not None:
      scheduler.step()

    perfmeasure, testloss,concat_labels, concat_pred, fnames  = evaluate_meanavgprecision(model, dataloader_test, criterion, device, numcl)
    testlosses.append(testloss)
    testperfs.append(perfmeasure)

    print('at epoch: ', epoch,' classwise perfmeasure ', perfmeasure)

    avgperfmeasure = np.mean(perfmeasure)
    print('at epoch: ', epoch,' avgperfmeasure ', avgperfmeasure)

    if avgperfmeasure > best_measure: #higher is better or lower is better?
      bestweights= model.state_dict()
      #TODO track current best performance measure and epoch
      best_measure = avgperfmeasure
      best_epoch = epoch
      best_c_pred = concat_pred
      best_c_lbls = concat_labels
      names = fnames
      #TODO save your scores

  return best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs, trainperfs, best_c_pred, best_c_lbls, names


class yourloss(nn.modules.loss._Loss):

    def __init__(self, reduction: str = 'mean') -> None:
        super(yourloss, self).__init__()
        #TODO
        self.num_classes = 17
        self.reduction = reduction


    def forward(self, input_: Tensor, target: Tensor) -> Tensor:
        #TODO
        #input_ = input_ + 1e-10
        off_target = 1-target
        log_pred = torch.log(input_)
        log_pred_1 = torch.log(1-input_)
        cross_entropy_all = target*log_pred + off_target*log_pred_1
        loss = -torch.sum(cross_entropy_all)/input_.shape[0]
        return loss


def runstuff():
  config = dict()
  config['use_gpu'] = True #True #TODO change this to True for training on the cluster
  config['lr'] = 0.01
  config['batchsize_train'] = 16
  config['batchsize_val'] = 34
  config['maxnumepochs'] = 30
  config['scheduler_stepsize'] = 5
  config['scheduler_factor'] = 0.3

  # This is a dataset property.
  config['numcl'] = 17
  seed = 20001
  if config['use_gpu'] == True:
      torch.cuda.manual_seed_all(seed)
  else:
      torch.manual_seed_all(seed)



  # Data augmentations.
  data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          ChannelSelect(channels=[0, 1, 2]),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          ChannelSelect(channels=[0, 1, 2]),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
  }

  dirname = os.getcwd()
  main_path = dirname + "/rainforest"
  gpu_path = "/itf-fi-ml/shared/IN5400/2022_mandatory1/"
  print(main_path)


  # Device
  if True == config['use_gpu']:
      device= torch.device('cuda:0')
      path = gpu_path
  else:
      device= torch.device('cpu')
      path = main_path
  # Datasets
  image_datasets={}
  #image_datasets['train']=dataset_voc(root_dir='/itf-fi-ml/shared/IN5400/dataforall/mandatory1/',trvaltest=0, transform=data_transforms['train'])
  #image_datasets['val']=dataset_voc(root_dir='/itf-fi-ml/shared/IN5400/dataforall/mandatory1/',trvaltest=1, transform=data_transforms['val'])
  image_datasets['train']=RainforestDataset(root_dir=path,trvaltest=0, transform=data_transforms['train'])
  image_datasets['val']=RainforestDataset(root_dir=path,trvaltest=1, transform=data_transforms['val'])
  # Dataloaders
  #TODO use num_workers=1
  dataloaders = {}
  dataloaders['train'] = DataLoader(image_datasets['train'], batch_size=config['batchsize_train'], shuffle=True)
  dataloaders['val'] = DataLoader(image_datasets['val'], batch_size=config['batchsize_val'], shuffle=False)

  # Model
  # TODO create an instance of the network that you want to use.
  pretrained_net = resnet18(pretrained=True)# TwoNetworks()
  model = SingleNetwork(pretrained_net)

  model = model.to(device)


  lossfct = yourloss()

  #TODO
  # Observe that all parameters are being optimized
  someoptimizer = torch.optim.Adam(params = model.parameters(), lr = config['lr'], betas = (0.9, 0.999))

  # Decay LR by a factor of 0.3 every X epochs
  #TODO
  somelr_scheduler = torch.optim.lr_scheduler.StepLR(someoptimizer, step_size=config['scheduler_stepsize'], gamma=config['scheduler_factor'])

  best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs, trainperfs, preds, labels, fnames = traineval2_model_nocv(dataloaders['train'], dataloaders['val'] ,  model ,  lossfct, someoptimizer, somelr_scheduler, num_epochs= config['maxnumepochs'], device = device , numcl = config['numcl'] )
  print("Best epoch: ", best_epoch)
  print("Best measure: ", best_measure)
  # Save best weights
  torch.save(bestweights, "./best_model.pt")
  # Write important outputs to file
  file_classes = open('classes_avg_scores.csv', 'w')
  class_writer = csv.writer(file_classes)
  header_row_classes = ['clear', 'cloudy', 'haze', 'partly_cloudy',
             'agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
             'blow_down', 'conventional_mine', 'cultivation', 'habitation',
             'primary', 'road', 'selective_logging', 'slash_burn', 'water']
  class_writer.writerow(header_row_classes)
  file_losses = open('losses_mAP.csv', 'w')
  loss_writer = csv.writer(file_losses)
  header_row_losses = ['training_loss','validation_loss','mean_average_score']
  loss_writer.writerow(header_row_losses)

  for epoch in range(config['maxnumepochs']):
      test_performances_at_epoch = testperfs[epoch]
      train_performances_at_epoch = trainperfs[epoch]
      mAP_test = np.mean(test_performances_at_epoch)
      mAP_train = np.mean(train_performances_at_epoch)
      class_writer.writerow(test_performances_at_epoch)
      performance_measures = []
      performance_measures.append(trainlosses[epoch])
      performance_measures.append(testlosses[epoch])
      performance_measures.append(mAP_test)
      performance_measures.append(mAP_train)
      loss_writer.writerow(performance_measures)

  file_classes.close()
  file_losses.close()
  file_fnames = open('fnames.csv', 'w')
  name_writer = csv.writer(file_fnames)
  name_writer.writerow(["filenames"])


  file_pred = open('predictions.csv', 'w')
  pred_writer = csv.writer(file_pred)
  pred_writer.writerow(header_row_classes)

  file_lbls = open('labels.csv', 'w')
  lbls_writer = csv.writer(file_lbls)
  lbls_writer.writerow(header_row_classes)

  for i in range(preds.shape[0]):
      pred_row = preds[i, :]
      lbls_row = labels[i, :]
      filename = fnames[i]
      pred_writer.writerow(pred_row)
      lbls_writer.writerow(lbls_row)
      name_writer.writerow([filename])

  file_pred.close()
  file_lbls.close()
  file_fnames.close()





if __name__=='__main__':

  runstuff()

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import PIL.Image
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from YourNetwork import SingleNetwork
from torchvision.models import resnet18


def get_classes_list():
    classes = ['clear', 'cloudy', 'haze', 'partly_cloudy',
               'agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
               'blow_down', 'conventional_mine', 'cultivation', 'habitation',
               'primary', 'road', 'selective_logging', 'slash_burn', 'water']
    return classes, len(classes)


class ChannelSelect(torch.nn.Module):
    """This class is to be used in transforms.Compose when you want to use selected channels. e.g only RGB.
    It works only for a tensor, not PIL object.
    Args:
        channels (list or int): The channels you want to select from the original image (4-channel).

    Returns: img
    """
    def __init__(self, channels=[0, 1, 2]):
        super().__init__()
        self.channels = channels
        self.num_channels = len(channels)

    def forward(self, img):
        """
        Args:
            img (Tensor): Image
        Returns:
            Tensor: Selected channels from the image.
        """
        return img[self.channels, ...]

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RainforestDataset(Dataset):
    def __init__(self, root_dir, trvaltest, transform):


        classes, num_classes = get_classes_list()

        self.transform = transform
        self.root_dir = root_dir
        self.data = root_dir + "/train_v2.csv"
        self.image_dir = root_dir + "/train-tif-v2/"
        self.ending = ".tif"

        dlabels = pd.read_csv(self.data)
        N = 5000
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
        labels = preprocessing.binarize(labels)
        img_filenames = dlabels["image_name"][0:N] + self.ending

        seed = 1001
        img_train, img_val, label_train, label_val= train_test_split(img_filenames, labels, test_size=0.33, random_state=seed)

        imgs_train = []; imgs_val = []
        labels_train = []; labels_val = []
        for img in img_train:
            imgs_train.append(img)
        for label in label_train:
            labels_train.append(label)
        for img in img_val:
            imgs_val.append(img)
        for label in label_val:
            labels_val.append(label)




        if trvaltest == 0:
            self.img_filenames = imgs_train
            self.labels = labels_train
        elif trvaltest == 1:
            self.img_filenames = imgs_val
            self.labels = labels_val
        else:
            print("trvaltest takes input 0 (train), 1 (validation).\n Try again.\n")
            exit()

        """
        # TODO Binarise your multi-labels from the string. HINT: There is a useful sklearn function to
        # help you binarise from strings.
	    dlabels = pd.read_csv(self.data)
	    N = 100
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
		          labels.append(labels)
	    self.labels = preprocessing.binarize(labels)

	    self.img_filenames = dlabels["image_name"]
        # TODO Perform a test train split. It's recommended to use sklearn's train_test_split with the following
        # parameters: test_size=0.33 and random_state=0 - since these were the parameters used
        # when calculating the image statistics you are using for data normalisation.
        #for debugging you can use a test_size=0.66 - this trains then faster
        seed = 0
        self.imgs_train, self.imgs_test, self.labels_train, self.labels_test = train_test_split(self.img_filenames, self.labels, test_size=0.33, random_state=seed)
        """


        # OR optionally you could do the test train split of your filenames and labels once, save them, and
        # from then onwards just load them from file.


    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        # TODO get the label and filename, and load the image from file.
        img_dir = self.image_dir + self.img_filenames[idx]
        img = PIL.Image.open(img_dir)
        labels = self.labels[idx]
        img_filename = self.img_filenames[idx]

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        sample = {'image': img,
                  'label': labels,
                  'filename': img_filename}
        """"
	    img_dir = self.image_dir + self.img_filenames[idx]
	    img = PIL.Image.open(img_dir)

	    labels = self.labels[idx]

	    if self.transform:
		     img = self.transform(img)
	        else:
		img = transforms.ToTensor()(img)

        sample = {'image': img,
                  'label': labels,
                  'filename': img_dir}
        """
        return sample


if __name__ == "__main__":
    main_path = "/home/jeb/Documents/IN5400/student_version/rainforest"
    trainvaltest = 1
    transform = transforms.Compose([
        transforms.ToTensor(),
        ChannelSelect(channels=[0,1,2]),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    )
    dataset = RainforestDataset(root_dir=main_path, trvaltest=trainvaltest, transform=transform)
    print(dataset[0]['filename'])
    print(dataset[0]['image'].shape)
    print(dataset.__len__())
    for i in range(dataset.__len__()):
        print(dataset[i]['filename'])

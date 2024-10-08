import os
import numpy as np
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex, OnMemoryImagePaths

class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example

class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file, random_crop=True, augment=True, gray=True):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=random_crop, augment=augment, gray=gray)

class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file, random_crop=False, augment=False, gray=True):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=random_crop, augment=augment, gray=gray)
        
class CustomOnMemoryTrain(CustomBase):
    def __init__(self, size, training_images_list_file, random_crop=True, augment=True, gray=True):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = OnMemoryImagePaths(paths=paths, size=size, random_crop=random_crop, augment=augment, gray=gray)

class CustomOnMemoryTest(CustomBase):
    def __init__(self, size, test_images_list_file, random_crop=False, augment=False, gray=True):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = OnMemoryImagePaths(paths=paths, size=size, random_crop=random_crop, augment=augment, gray=gray)
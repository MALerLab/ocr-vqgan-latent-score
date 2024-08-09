import bisect
import numpy as np
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None, augment=False, gray=True):
        self.size = size
        self.random_crop = random_crop
        self.augment = augment
        self.gray = gray
        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = A.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = A.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = A.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = A.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

        if self.augment:
            # Add data aug transformations
            if self.gray:
                self.data_augmentation = A.Compose([
                    A.GaussianBlur(p=0.1),
                ])
            else:
                self.data_augmentation = A.Compose([
                    A.GaussianBlur(p=0.1),
                    A.OneOf([
                        A.HueSaturationValue (p=0.3),
                        A.ToGray(p=0.3),
                        A.ChannelShuffle(p=0.3)
                    ], p=0.3)
                ])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if self.gray:
            if not image.mode == "L":
                image = image.convert("L")
        else:
            if not image.mode == "RGB": 
                image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        if self.augment:
            image = self.data_augmentation(image=image)['image']
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example

class OnMemoryImagePaths(ImagePaths):
    def __init__(self, paths, size=None, random_crop=False, labels=None, augment=False, gray=True):
        super().__init__(paths, size, random_crop, labels, augment, gray)
        self.images = self.load_images(self.labels["file_path_"])
        
    def load_images(self, paths):
        images = []
        print("Loading Images on the Memory...")
        for path in tqdm(paths):
            image = Image.open(path)
            if self.gray:
                if not image.mode == "L":
                    image = image.convert("L")
            else:
                if not image.mode == "RGB": 
                    image = image.convert("RGB")
            image = np.array(image).astype(np.uint8)
            images.append(image)
        return images
    
    def preprocess_image(self, image):
        image = self.preprocessor(image=image)["image"]
        if self.augment:
            image = self.data_augmentation(image=image)['image']
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
    
    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.images[i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example
    
class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

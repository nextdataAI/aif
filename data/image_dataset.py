import os
import PIL.Image as im
import numpy as np
from tqdm import tqdm

__all__ = ["ImageDataset"]


class ImageDataset:
    def __init__(self, max_data: int = 1000000):
        super().__init__()
        self.images = []
        self.labels = []
        max_data_local = len(os.listdir('data/maze_images_dataset'))
        for img in tqdm(os.listdir('data/maze_images_dataset'), desc='Loading dataset', unit='Images',
                        total=max_data_local if max_data>max_data_local else max_data):
            self.images.append(np.array(im.open(f'data/maze_images_dataset/{img}'), dtype=np.uint8))
            self.labels.append(int(img.split('_')[-1].split('.')[0]))
            if len(self.images) >= max_data:
                break
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.images)

    def __call__(self):
        return self.images, self.labels

    def __repr__(self):
        return f'Image Dataset with {len(self)} images'

    def __add__(self, other):
        return np.concatenate((self.images, other.images)), np.concatenate((self.labels, other.labels))

    def __iadd__(self, other):
        self.images = np.concatenate((self.images, other.images))
        self.labels = np.concatenate((self.labels, other.labels))
        return self

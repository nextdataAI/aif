import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
__all__ = ["ImageDataset", "AsciiDataset", 'get_player_floor_target_png']

import PIL.Image as im


def get_player_floor_target_png(all_val=False):
    player = np.array(im.open(f'{os.path.dirname(__file__)}/minihack_images/player.png'), dtype=np.uint8)
    floor = np.array(im.open(f'{os.path.dirname(__file__)}/minihack_images/floor.png'), dtype=np.uint8)
    target = np.array(im.open(f'{os.path.dirname(__file__)}/minihack_images/target.png'), dtype=np.uint8)
    floor_red = np.array(im.open(f'{os.path.dirname(__file__)}/minihack_images/floor_red.png'), dtype=np.uint8)
    floor_yellow = np.array(im.open(f'{os.path.dirname(__file__)}/minihack_images/floor_yellow.png'), dtype=np.uint8)
    flood_blue = np.array(im.open(f'{os.path.dirname(__file__)}/minihack_images/floor_blue.png'), dtype=np.uint8)
    if all_val:
        return player, floor, floor_red, floor_yellow, flood_blue, target
    return player, floor, target


class ImageDataset:
    def __init__(self, max_data=None, kind: str = 'train'):
        super().__init__()
        self.images = []
        self.labels = []
        self.base_path = f'{os.path.dirname(__file__)}/maze_images_dataset/{kind}/'
        df = pd.read_csv(f'{os.path.dirname(__file__)}/{kind}_data.csv')

        # Process file paths and labels
        df['file_name'] = self.base_path + df['file_name'] + '.png'
        df['label'] = df['label'].astype(str)
        if max_data is not None:
            df = df.sample(max_data)
        # Create your generators
        self.image_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        self.image_generator = self.image_datagen.flow_from_dataframe(
            df,
            x_col='file_name',
            y_col='label',
            target_size=(1264, 1264),
            batch_size=32,
            class_mode='sparse',
            shuffle=True
        )

    def __call__(self, *args, **kwargs):
        return self.image_generator


class AsciiDataset:
    def __init__(self, max_data=None, kind: str = 'train'):
        super().__init__()
        self.maps = []
        self.labels = []
        self.base_path = f'{os.path.dirname(__file__)}/maze_ascii_dataset/{kind}/'
        df = pd.read_csv(f'{os.path.dirname(__file__)}/{kind}_data.csv')  # Load your DataFrame

        # Process file paths and labels
        if max_data is not None:
            df = df.sample(max_data, random_state=42)
        df['file_name'] = self.base_path + df['file_name'] + '.npy'  # Adjust the path to your images
        df['label'] = df['label'].astype(int)
        df['matrix'] = df['file_name'].apply(lambda x: np.load(x))
        self.maps = np.array(df['matrix'])
        self.labels = df['label'].to_numpy(dtype='int32')

    def __call__(self, *args, **kwargs):
        self.maps = np.array(value.ravel() / 255 for value in self.maps)
        return self.maps, self.labels

    def __call_2__(self):
        new_maps = []
        new_labels = []
        for elem, label in zip(self.maps, self.labels):
            start = np.where(elem == ord('@'))
            end = np.where(elem == ord('>'))
            elem = np.array([start[0], start[1], end[0], end[1]])
            new_maps.append(elem)
            new_labels.append((abs(start[0] - end[0]) * 2 + abs(start[1] - end[1]) * 2))
        self.maps = np.array(new_maps)
        self.labels = np.array(new_labels)
        return self.maps, self.labels

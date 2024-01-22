import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
__all__ = ["ImageDataset", "AsciiDataset"]


class ImageDataset:
    def __init__(self, max_data=None, kind: str = 'train'):
        super().__init__()
        self.images = []
        self.labels = []
        self.base_path = f'data/maze_images_dataset/{kind}/'
        df = pd.read_csv(f'data/{kind}_data.csv')

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
        self.base_path = f'data/maze_ascii_dataset/{kind}/'
        df = pd.read_csv(f'data/{kind}_data.csv')  # Load your DataFrame

        # Process file paths and labels
        if max_data is not None:
            df = df.sample(max_data, random_state=42)
        df['file_name'] = self.base_path + df['file_name'] + '.npy'  # Adjust the path to your images
        df['label'] = df['label'].astype(int)
        df['matrix'] = df['file_name'].apply(lambda x: np.load(x))
        self.maps = np.array(df['matrix'])
        self.labels = df['label'].to_numpy(dtype='int32')

    def __call__(self, *args, **kwargs):
        self.maps = np.array(value.ravel()/255 for value in self.maps)
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

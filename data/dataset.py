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
        df['file_name'] = self.base_path + df['file_name'] + '.ascii'  # Adjust the path to your images
        df['label'] = df['label'].astype(int)
        df['map'] = df['file_name'].apply(lambda x: open(x).read())
        self.maps = df['map'].to_numpy(dtype='int')
        self.labels = df['label'].to_numpy(dtype='float32')

    def __call__(self, *args, **kwargs):
        return self.maps, self.labels

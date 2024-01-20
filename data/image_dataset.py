import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

__all__ = ["ImageDataset"]


class ImageDataset:
    def __init__(self, max_data=None, kind: str = 'train'):
        super().__init__()
        self.images = []
        self.labels = []
        self.base_path = f'data/maze_images_dataset/{kind}/'
        df = pd.read_csv(f'data/maze_images_dataset/{kind}_data.csv')  # Load your DataFrame

        # Process file paths and labels
        df['file_name'] = self.base_path + df['file_name']  # Adjust the path to your images
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
        # self.image_generator = self.image_datagen.flow_from_directory(
        #     self.base_path,
        #     classes=self.labels,
        #     target_size=(400, 400),
        #     batch_size=32,
        #     class_mode="sparse",
        # )
        self.image_generator = self.image_datagen.flow_from_dataframe(
            df,
            x_col='file_name',
            y_col='label',
            target_size=(400, 400),  # Adjust to match your model's input size
            batch_size=32,
            class_mode='sparse',  # Change to 'binary' if you have binary classes
            shuffle=True
        )

    def __call__(self, *args, **kwargs):
        return self.image_generator

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

TOTAL_IMAGE_PIXELS = 784

class DigitDataset(Dataset):
    def __init__(self, file_path, normalise_pixels=False):
        self.df = pd.read_csv(file_path)
        self.normalise_pixels = normalise_pixels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = row.get("label")
        pixels = self.get_pixels_tensor(row)

        # The submission dataset won't have a label column, so we return a placeholder label of -1
        if label is None:
            label = -1

        return (pixels, label)

    def get_pixels_tensor(self, row):
        flattened_pixel_values = []
        for pixel_index in range(TOTAL_IMAGE_PIXELS):
            column_name = f"pixel{pixel_index}"
            pixel_value = row[column_name]
            flattened_pixel_values.append(pixel_value)

        pixel_tensor = torch.tensor(flattened_pixel_values)
        pixel_tensor = pixel_tensor.reshape(1, 28, 28)

        if self.normalise_pixels:
            pixel_tensor = pixel_tensor / 255

        return pixel_tensor
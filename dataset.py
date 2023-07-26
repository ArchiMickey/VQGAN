from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor, CenterCrop
import os
import numpy as np
from loguru import logger


class ImageDataset(Dataset):
    def __init__(self, paths, img_size=None):
        super(ImageDataset, self).__init__()

        self.paths = paths
        self.images = []
        for path in self.paths:
            logger.info(f"Loading images from {path}...")
            images_in_path = os.listdir(path)
            self.images += [os.path.join(path, image) for image in images_in_path]
        self.images.sort()

        self.img_size = img_size
        self.transform = Compose(
            [CenterCrop(img_size), ToTensor(), Normalize((0.5,), (0.5,))]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transform(image)
        return image

    def get_dataloader(self, batch_size, shuffle=True, num_workers=4):
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )


if __name__ == "__main__":
    from icecream import install

    install()

    paths = ["./data/flower"]
    dataset = ImageDataset(paths, 256)
    ic(dataset[0])

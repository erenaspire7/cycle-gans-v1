from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class GANDataset(Dataset):
    def __init__(self, root_human, root_statue, transform=None):
        self.root_human = root_human
        self.root_statue = root_statue
        self.transform = transform

        self.human_images = os.listdir(root_human)
        self.statue_images = os.listdir(root_statue)
        self.length_dataset = max(len(self.human_images), len(self.statue_images))
        self.human_len = len(self.human_images)
        self.statue_len = len(self.statue_images)

    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        human_img = self.human_images[index % self.human_len]
        statue_img = self.statue_images[index % self.statue_len]

        human_path = os.path.join(self.root_human, human_img)
        statue_path = os.path.join(self.root_statue, statue_img)

        human_img = np.array(Image.open(human_path).convert("RGB"))
        statue_img = np.array(Image.open(statue_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=human_img, image0=statue_img)
            human_img = augmentations["image"]
            statue_img = augmentations["image0"]

        return human_img, statue_img
import os
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class CLICDataset(Dataset):
    def __init__(self, data_root, patch_size):
        self.data_root = data_root
        self.patch_size = patch_size
        self.image_paths = glob.glob(os.path.join(self.data_root, "*.png"))
        self.transform = transforms.Compose([
            transforms.RandomCrop(self.patch_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)

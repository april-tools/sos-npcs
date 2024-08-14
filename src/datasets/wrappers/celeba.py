from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision.transforms import transforms


def rgb2ycc(rgb_image):
    assert rgb_image.size(0) == 3

    def forward_lift(x, y):
        diff = (y - x) % 256
        average = (x + (diff >> 1)) % 256
        return average, diff

    red, green, blue = rgb_image[0], rgb_image[1], rgb_image[2]
    temp, co = forward_lift(red, blue)
    y, cg = forward_lift(green, temp)
    ycc_image = torch.stack([y, cg, co], dim=0)
    return ycc_image


def ycc2rgb(ycc_image):
    assert ycc_image.size(0) == 3

    def reverse_lift(average, diff):
        x = (average - (diff >> 1)) % 256
        y = (x + diff) % 256
        return x, y

    y, cg, co = ycc_image[0], ycc_image[1], ycc_image[2]
    green, temp = reverse_lift(y, cg)
    red, blue = reverse_lift(temp, co)
    rgb_image = torch.stack([red, green, blue], dim=0)
    return rgb_image


class CELEBA(Dataset):
    def __init__(self, root, split="all", ycc: Optional[bool] = False):
        if ycc:
            ts = [
                transforms.CenterCrop((140, 140)),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 255).to(torch.int64)),
                transforms.Lambda(rgb2ycc)
            ]
        else:
            ts = [
                transforms.CenterCrop((140, 140)),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 255).to(torch.int64)),
            ]
        transform = transforms.Compose(ts)

        self.celeba_dataset = CelebA(
            root=root, split=split, transform=transform, download=False
        )

    def __len__(self):
        return len(self.celeba_dataset)

    def __getitem__(self, idx):
        image, _ = self.celeba_dataset[idx]
        return image.view(image.shape[0], -1)

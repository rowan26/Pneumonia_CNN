from pathlib import Path
from typing import Callable
from torch.utils.data import Dataset
import skimage.io
import torch


class ChestXrayDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, int]], transform: Callable):
        self.pairs=pairs
        self.transform=transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label=self.pairs[index]
        img = skimage.io.imread(path)
        img = self.transform(img)
        img = torch.from_numpy(img)
        label = torch.tensor(label)
        return img, label
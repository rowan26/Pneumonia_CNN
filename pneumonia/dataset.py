from pathlib import Path
from typing import Callable
from torch.utils.data import Dataset, DataLoader
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



def get_dataloaders(train_dataset: Dataset, val_dataset: Dataset, batch_size: int=16) -> tuple[DataLoader, DataLoader]:

    """Construit les DataLoader train et val : train mélangé à chaque epoch,
    val dans un ordre fixe (aucun bénéfice à le mélanger, le modèle n'apprend
    pas pendant l'évaluation).
    """

    train_dataloader=DataLoader(train_dataset,batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader=DataLoader(val_dataset,batch_size=batch_size, shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader


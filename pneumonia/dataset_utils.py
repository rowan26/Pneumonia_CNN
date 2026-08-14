from pneumonia.input_validation import is_valid_extension
from pathlib import Path
import random
from collections import Counter
import torch

def count_images(path: Path) -> int:
    """Compte les fichiers valides (extension autorisée) dans un dossier."""

    count=0
    for element in path.iterdir():
        if element.is_file() and is_valid_extension(element.name):
            count+=1
    return count


def get_dataset_counts(data_dir: Path, 
                       splits: tuple[str,...]=("train","val","test"), 
                       classes: tuple[str,...]=("NORMAL","PNEUMONIA")) -> dict:
    
    """Compte les images valides pour chaque combinaison split/classe.

    Returns:
        Un dictionnaire imbriqué {split: {classe: nombre_d_images}}.
    """
    counts={}
    for split in splits:
        counts[split]={}
        for classe in classes:
            path=data_dir / split / classe
            counts[split][classe]=count_images(path)
    return counts


def display_dataset_counts(counts: dict) -> None:
    """Affiche un tableau récapitulatif des comptages par split, avec total."""

    classes = list(next(iter(counts.values())).keys())
    print("Split | " + " | ".join(classes) + " | Total")
    for split in counts:
        total=0
        values = []
        for classe in counts[split]:
            total+=counts[split][classe]
            values.append(str(counts[split][classe]))
        print(f"{split} | " + " | ".join(values) + f" | {total}")


def list_image_paths(path: Path) -> list[Path]:

    """Liste les chemins des fichiers image valides (extension autorisée) dans un dossier."""

    image_paths=[]
    for element in path.iterdir():
        if element.is_file() and is_valid_extension(element.name):
            image_paths.append(element)
    return image_paths

def stratified_train_val_split(
        normal_paths: list[Path], 
        pneumonia_paths: list[Path], 
        seed: int=42,
        val_size: float=0.1,) -> tuple[list[tuple[Path, int]], list[tuple[Path,int]]]:
    
    """Découpe les chemins NORMAL et PNEUMONIA en train/val, séparément par classe.

    Le split est stratifié : chaque classe est mélangée et découpée
    indépendamment, ce qui préserve dans train et val la même proportion
    NORMAL/PNEUMONIA que dans le dataset d'origine. Déterministe : la même
    seed produit toujours le même découpage.

    Args:
        normal_paths: Chemins des images de la classe NORMAL (label 0).
        pneumonia_paths: Chemins des images de la classe PNEUMONIA (label 1).
        seed: Graine du générateur aléatoire, pour la reproductibilité.
        val_size: Proportion de chaque classe allouée à val (ex. 0.1 = 10 %).

    Returns:
        Un tuple (train_pairs, val_pairs), chacun étant une liste de
        tuples (chemin, label).
    """
    rng=random.Random(seed)

    copie_normal=normal_paths.copy()
    rng.shuffle(copie_normal)

    len_copie=len(copie_normal)
    size=int(val_size * len_copie)

    normal_val = copie_normal[:size]
    normal_train = copie_normal[size:]

    copie_pneumonia=pneumonia_paths.copy()
    rng.shuffle(copie_pneumonia)

    len_copie=len(copie_pneumonia)
    size=int(val_size * len_copie)

    pneumonia_val = copie_pneumonia[:size]
    pneumonia_train = copie_pneumonia[size:]

    normal_val_labeled=[(path,0) for path in normal_val]
    normal_train_labeled=[(path,0) for path in normal_train]

    pneumonia_val_labeled=[(path,1) for path in pneumonia_val]
    pneumonia_train_labeled=[(path,1) for path in pneumonia_train]

    train_pairs=normal_train_labeled + pneumonia_train_labeled
    val_pairs=normal_val_labeled + pneumonia_val_labeled

    return train_pairs, val_pairs


def compute_class_weights(train_pairs: list[tuple[Path,int]]) -> dict:

    """Calcule un poids par classe, inversement proportionnel à sa fréquence.

    Destiné à être utilisé dans la fonction de loss pendant l'entraînement,
    pour compenser le déséquilibre entre classes sans modifier le
    chargement des données.
    """
    
    train_counts=Counter(label for _,label in train_pairs)
    sum_total=len(train_pairs)
    num_classes=len(train_counts)
    return {label: sum_total / (num_classes * count) for label, count in train_counts.items()}


def class_weights_to_tensor(weights: dict) -> torch.Tensor:
    
    """Convertit un dictionnaire {label: poids} en tensor, trié par label
    croissant, pour être utilisable directement comme paramètre `weight`
    d'une fonction de loss PyTorch (ex. CrossEntropyLoss).
    """

    sorted_dict=sorted(weights.items())
    weights_only=[weight for _,weight in sorted_dict]
    tensor=torch.tensor(weights_only)

    return tensor
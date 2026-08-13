from pneumonia.input_validation import is_valid_extension
from pathlib import Path


def count_images(path: Path) -> int:
    """Compte les fichiers valides (extension autorisée) dans un dossier."""

    count=0
    for element in path.iterdir():
        if element.is_file() and is_valid_extension(element.name):
            count+=1
    return count


def get_dataset_counts(data_dir: Path, splits: tuple[str,...]=("train","val","test"), classes: tuple[str,...]=("NORMAL","PNEUMONIA") ) -> dict:
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
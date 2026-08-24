from pathlib import Path
from collections import Counter
from pneumonia.dataset_utils import list_image_paths, stratified_train_val_split, display_dataset_counts, compute_class_weights


from pneumonia.config import LABEL_NAMES

def pairs_to_counts(train_pairs, val_pairs) -> dict:

    """Convertit des paires (chemin, label) en comptages {split: {classe: nombre}}."""
    train_counts=Counter(label for _, label in train_pairs)
    val_counts=Counter(label for _,label in val_pairs)

    return {
        "train": {LABEL_NAMES[label]: count for label, count in train_counts.items()},
        "val": {LABEL_NAMES[label]: count for label, count in val_counts.items()},
    }

def main() -> None:

    project_root = Path(__file__).resolve().parent.parent
    data_dir= project_root / "data" / "chest_xray"
    
    normal_paths = list_image_paths(data_dir / "train" / "NORMAL")
    pneumonia_paths = list_image_paths(data_dir / "train" / "PNEUMONIA")

    train_pairs, val_pairs = stratified_train_val_split(normal_paths, pneumonia_paths)
    dataset_counts = pairs_to_counts(train_pairs, val_pairs)
    display_dataset_counts(dataset_counts)
    weights=compute_class_weights(train_pairs)
    print(weights)

if __name__ == "__main__":
    main()
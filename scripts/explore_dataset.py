from pneumonia.dataset_utils import get_dataset_counts, display_dataset_counts
from pathlib import Path

def main() -> None:
    """Affiche un récapitulatif du nombre d'images par split et par classe."""
    project_root = Path(__file__).resolve().parent.parent
    data_dir= project_root / "data" / "chest_xray"

    counts=get_dataset_counts(data_dir)
    display_dataset_counts(counts)

if __name__ == "__main__":
    main()
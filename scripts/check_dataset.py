from pathlib import Path
from pneumonia.dataset_utils import list_image_paths, stratified_train_val_split
from pneumonia.dataset import ChestXrayDataset
from pneumonia.transforms import preprocess_xray


def main() -> None:

    project_root = Path(__file__).resolve().parent.parent
    data_dir= project_root / "data" / "chest_xray"
    
    normal_paths = list_image_paths(data_dir / "train" / "NORMAL")
    pneumonia_paths = list_image_paths(data_dir / "train" / "PNEUMONIA")

    train_pairs,_ = stratified_train_val_split(normal_paths, pneumonia_paths) # _ is the ignore value
    dataset= ChestXrayDataset(train_pairs, preprocess_xray)
    img,label=dataset[0]
    print(img.shape,label)

if __name__ == "__main__":
    main()
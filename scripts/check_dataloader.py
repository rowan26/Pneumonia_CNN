from pathlib import Path
from pneumonia.dataset_utils import list_image_paths, stratified_train_val_split
from pneumonia.dataset import ChestXrayDataset, get_dataloaders
from pneumonia.transforms import preprocess_xray


def main() -> None:

    project_root = Path(__file__).resolve().parent.parent
    data_dir= project_root / "data" / "chest_xray"
    
    normal_paths = list_image_paths(data_dir / "train" / "NORMAL")
    pneumonia_paths = list_image_paths(data_dir / "train" / "PNEUMONIA")

    train_pairs, val_pairs = stratified_train_val_split(normal_paths, pneumonia_paths) # _ is the ignore value
    train_dataset= ChestXrayDataset(train_pairs, preprocess_xray)
    val_dataset= ChestXrayDataset(val_pairs, preprocess_xray)
    train_dataloader, val_dataloader=get_dataloaders(train_dataset, val_dataset)
    first_batch_train=next(iter(train_dataloader))
    first_batch_val=next(iter(val_dataloader))
    img_train, label_train = first_batch_train
    img_val, label_val = first_batch_val
    print(img_train.shape, label_train.shape)
    print(img_val.shape, label_val.shape)

if __name__ == "__main__":
    main()
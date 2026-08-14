from pathlib import Path
import torch
import torch.nn as nn
from pneumonia.dataset_utils import list_image_paths, stratified_train_val_split, compute_class_weights, class_weights_to_tensor
from pneumonia.dataset import ChestXrayDataset, get_dataloaders
from pneumonia.transforms import preprocess_xray
from pneumonia.model_loader import load_model
from pneumonia.model_utils import adapt_model_head
from pneumonia.training import train_model


def main(num_epochs: int=1, learning_rate: float=1e-4) -> None:

    project_root = Path(__file__).resolve().parent.parent
    data_dir= project_root / "data" / "chest_xray"
    
    normal_paths = list_image_paths(data_dir / "train" / "NORMAL")
    pneumonia_paths = list_image_paths(data_dir / "train" / "PNEUMONIA")

    train_pairs, val_pairs = stratified_train_val_split(normal_paths, pneumonia_paths)

    train_dataset= ChestXrayDataset(train_pairs, preprocess_xray)
    val_dataset= ChestXrayDataset(val_pairs, preprocess_xray)

    train_dataloader, val_dataloader=get_dataloaders(train_dataset, val_dataset)
    
    model=load_model("densenet121-res224-all")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")
    
    model_adapted=adapt_model_head(model)
    model_adapted=model_adapted.to(device)

    weights=compute_class_weights(train_pairs)
    weights_to_tensor=class_weights_to_tensor(weights)

    optimizer=torch.optim.Adam(model_adapted.parameters(), lr=learning_rate)

    loss=nn.CrossEntropyLoss(weight=weights_to_tensor)

    train_model(model_adapted, train_dataloader, val_dataloader, loss, optimizer, device, num_epochs=num_epochs)


if __name__ == "__main__":
    main()
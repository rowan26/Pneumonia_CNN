from pathlib import Path
import torch
import torch.nn as nn
from pneumonia.dataset_utils import list_image_paths, stratified_train_val_split, compute_class_weights, class_weights_to_tensor
from pneumonia.dataset import ChestXrayDataset, get_dataloaders
from pneumonia.transforms import preprocess_xray
from pneumonia.model_loader import load_model
from pneumonia.model_utils import adapt_model_head
from pneumonia.training import train_model
import mlflow


def main(
        data_dir: Path | None = None,
        num_epochs: int = 1,
        learning_rate: float = 1e-4,
        batch_size: int=16,
        val_size: float=0.1,
        checkpoint_path: Path | None = None,
        seed: int = 42,
        weights_name: str="densenet121-res224-all",
        tracking_uri: str | None = None) -> None:

    torch.manual_seed(seed)

    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    with mlflow.start_run() as run:
        run_id=run.info.run_id

        mlflow.log_param("learning_rate",learning_rate)
        mlflow.log_param("num_epochs",num_epochs)
        mlflow.log_param("batch_size",batch_size)
        mlflow.log_param("val_size",val_size)
        mlflow.log_param("seed",seed)
        mlflow.log_param("model",weights_name)
       
        project_root = Path(__file__).resolve().parent.parent

        if data_dir is None:
            data_dir = project_root / "data" / "chest_xray"

        if checkpoint_path is None:
            checkpoint_path = project_root / "artifacts" / "best_model.pth"

        normal_paths = list_image_paths(data_dir / "train" / "NORMAL")
        pneumonia_paths = list_image_paths(data_dir / "train" / "PNEUMONIA")

        train_pairs, val_pairs = stratified_train_val_split(normal_paths, pneumonia_paths, val_size=val_size,seed=seed)

        train_dataset = ChestXrayDataset(train_pairs, preprocess_xray)
        val_dataset = ChestXrayDataset(val_pairs, preprocess_xray)

        train_dataloader, val_dataloader = get_dataloaders(train_dataset, val_dataset, batch_size=batch_size)

        model = load_model(weights_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device utilisé : {device}")

        mlflow.log_param("device", str(device))

        mlflow.log_param("n_train_images", len(train_pairs))
        mlflow.log_param("n_val_images", len(val_pairs))


        model_adapted = adapt_model_head(model)
        model_adapted = model_adapted.to(device)

        weights = compute_class_weights(train_pairs)
        weights_to_tensor = class_weights_to_tensor(weights)
        weights_to_tensor = weights_to_tensor.to(device)

        optimizer = torch.optim.Adam(model_adapted.parameters(), lr=learning_rate)
        loss = nn.CrossEntropyLoss(weight=weights_to_tensor)

        checkpoint_path = checkpoint_path.parent / f"best_model_{run_id[:8]}.pth"
        mlflow.log_param("checkpoint_file", checkpoint_path.name)

        history=train_model(model_adapted, train_dataloader, val_dataloader, loss, optimizer, device,
                    num_epochs=num_epochs, checkpoint_path=checkpoint_path)

        for epoch in range(len(history["train_loss"])):
            mlflow.log_metric("train_loss", history["train_loss"][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history["train_accuracy"][epoch], step=epoch)
            mlflow.log_metric("val_loss", history["val_loss"][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history["val_accuracy"][epoch], step=epoch)

if __name__ == "__main__":
    main()
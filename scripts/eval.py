from pathlib import Path
import torch
import torch.nn as nn
from pneumonia.dataset_utils import list_image_paths, build_pairs
from pneumonia.evaluation import collect_predictions, compute_metrics
from pneumonia.dataset import ChestXrayDataset, get_dataloaders
from pneumonia.transforms import preprocess_xray
from pneumonia.model_loader import load_finetuned_model
import mlflow
from torch.utils.data import DataLoader

from pneumonia.config import DEFAULT_WEIGHTS_NAME

def main(
        data_dir: Path | None = None,
        batch_size: int=16,
        checkpoint_path: Path | None = None,
        run_id: str | None = None,
        weights_name: str = DEFAULT_WEIGHTS_NAME,
        tracking_uri: str | None = None) -> None:
        
       
    project_root = Path(__file__).resolve().parent.parent

    if data_dir is None:
        data_dir = project_root / "data" / "chest_xray"

    if checkpoint_path is None:
        checkpoint_path = project_root / "artifacts" / "best_model.pth"

    normal_paths = list_image_paths(data_dir / "test" / "NORMAL")
    pneumonia_paths = list_image_paths(data_dir / "test" / "PNEUMONIA")

    test_pairs = build_pairs(normal_paths, pneumonia_paths)

    test_dataset = ChestXrayDataset(test_pairs, preprocess_xray)
    test_dataloader= DataLoader(test_dataset,batch_size=batch_size, shuffle=False, drop_last=False)

    model = load_finetuned_model(checkpoint_path, weights_name=weights_name, num_classes=2, device="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    labels, predictions, probabilities = collect_predictions(model, test_dataloader, device=device)
    metrics=compute_metrics(labels, predictions, probabilities)


            # Affichage des métriques
    print(f"\nÉvaluation sur le test set ({len(test_pairs)} images)")
    print(f"  Recall (PNEUMONIA)  : {metrics['recall']:.4f}")
    print(f"  Precision           : {metrics['precision']:.4f}")
    print(f"  F1-score            : {metrics['f1']:.4f}")
    print(f"  AUC-ROC             : {metrics['roc_auc']:.4f}")
    print("\nMatrice de confusion")
    print(f"  Vrais négatifs  (NORMAL bien classés)     : {metrics['true_negatives']}")
    print(f"  Faux positifs   (NORMAL vus PNEUMONIA)    : {metrics['false_positives']}")
    print(f"  Faux négatifs   (PNEUMONIA ratés)         : {metrics['false_negatives']}")
    print(f"  Vrais positifs  (PNEUMONIA bien détectés) : {metrics['true_positives']}")

    # Logging MLflow dans le run d'entraînement existant, si demandé
    if run_id is not None:
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        with mlflow.start_run(run_id=run_id):
            for name, value in metrics.items():
                mlflow.log_metric(f"test_{name}", value)
        print(f"\nMétriques loggées dans le run MLflow {run_id}")


    return

if __name__ == "__main__":
    main()
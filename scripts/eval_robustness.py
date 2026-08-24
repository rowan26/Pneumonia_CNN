from pathlib import Path
import torch
from torch.utils.data import DataLoader
import mlflow

from pneumonia.dataset_utils import list_image_paths, build_pairs
from pneumonia.evaluation import collect_predictions, compute_metrics
from pneumonia.dataset import ChestXrayDataset
from pneumonia.perturbations import add_gaussian_noise, apply_luminance_scaling
from pneumonia.transforms import preprocess_xray,compose_with_preprocess
from pneumonia.model_loader import load_finetuned_model

from pneumonia.config import DEFAULT_WEIGHTS_NAME

def main(
        data_dir: Path | None = None,
        batch_size: int = 16,
        checkpoint_path: Path | None = None,
        run_id: str | None = None,
        weights_name: str = DEFAULT_WEIGHTS_NAME,
        tracking_uri: str | None = None) -> None:

    project_root = Path(__file__).resolve().parent.parent

    if data_dir is None:
        data_dir = project_root / "data" / "chest_xray"
    if checkpoint_path is None:
        checkpoint_path = project_root / "artifacts" / "best_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    normal_paths = list_image_paths(data_dir / "test" / "NORMAL")
    pneumonia_paths = list_image_paths(data_dir / "test" / "PNEUMONIA")
    test_pairs = build_pairs(normal_paths, pneumonia_paths)

    model = load_finetuned_model(checkpoint_path, weights_name=weights_name, num_classes=2, device=str(device))
    model = model.to(device)

    conditions = {
        "baseline": preprocess_xray,
        "bruit gaussien (sigma=10)": compose_with_preprocess(add_gaussian_noise),
        "luminosite (x1.2)": compose_with_preprocess(apply_luminance_scaling),
    }

    all_metrics = {}

    print(f"\nEvaluation de robustesse sur le test set ({len(test_pairs)} images)\n")
    print(f"{'Condition':<28} {'Recall':>8} {'Precision':>10} {'F1':>8} {'AUC':>8} {'Rates':>7} {'Fausses alertes':>17}")
    print("-" * 90)

    for name, transform in conditions.items():
        dataset = ChestXrayDataset(test_pairs, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        labels, preds, probs = collect_predictions(model, dataloader, device)
        metrics = compute_metrics(labels, preds, probs)
        all_metrics[name] = metrics

        print(f"{name:<28} {metrics['recall']:>8.4f} {metrics['precision']:>10.4f} "
              f"{metrics['f1']:>8.4f} {metrics['roc_auc']:>8.4f} "
              f"{metrics['false_negatives']:>7} {metrics['false_positives']:>17}")

    if run_id is not None:
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        with mlflow.start_run(run_id=run_id):
            for condition_name, metrics in all_metrics.items():
                prefix = condition_name.split(" ")[0]
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"test_{prefix}_{metric_name}", value)
        print(f"\nMetriques loggees dans le run MLflow {run_id}")


if __name__ == "__main__":
    main()
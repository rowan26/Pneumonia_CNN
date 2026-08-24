from pathlib import Path

from pneumonia.input_validation import is_valid_extension, is_valid_image
from pneumonia.model_loader import load_finetuned_model
from pneumonia.predict import predict

import argparse



def main() -> None:
    """Prédit la classe d'une radiographie thoracique passée en argument."""

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Chemin vers l'image à prédire")
    args = parser.parse_args()
    image_path = Path(args.image)

    if not is_valid_extension(image_path.name):
        print(f"Extension non supportée : {image_path.name}")
        return
    if not is_valid_image(image_path):
        print(f"Fichier illisible ou corrompu : {image_path}")
        return

    project_root = Path(__file__).resolve().parent.parent
    checkpoint_path = project_root / "artifacts" / "best_model.pth"

    model = load_finetuned_model(checkpoint_path)
    results = predict(model, image_path)

    print(f"Prédiction : {results['class_name']} (confiance : {results['confidence']:.2%})")

if __name__ == "__main__":
    main()
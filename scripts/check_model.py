from pneumonia.model_loader import load_model, load_finetuned_model
from pneumonia.model_utils import adapt_model_head
from pathlib import Path

from pneumonia.config import DEFAULT_WEIGHTS_NAME, NUM_CLASSES
def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    checkpoint_path = project_root / "artifacts" / "best_model.pth"

    
    model=load_model(DEFAULT_WEIGHTS_NAME)
    for name, module in model.named_children():
        print(name, "->", module)

    model = load_model()
    print("Avant :", model.classifier)
    
    model = adapt_model_head(model, num_classes=NUM_CLASSES)
    print("Après :", model.classifier)

    finetuned = load_finetuned_model(checkpoint_path)
    print("Modèle fine-tuné chargé :", finetuned.classifier)

    
if __name__ == "__main__":
    main()
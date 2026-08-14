from pneumonia.model_loader import load_model
from pneumonia.model_utils import adapt_model_head


def main() -> None:
    model=load_model("densenet121-res224-all")
    for name, module in model.named_children():
        print(name, "->", module)

    model = load_model()
    print("Avant :", model.classifier)
    
    model = adapt_model_head(model, num_classes=2)
    print("Après :", model.classifier)

if __name__ == "__main__":
    main()
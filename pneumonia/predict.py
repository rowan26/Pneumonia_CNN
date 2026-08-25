import skimage
import torch
from typing import BinaryIO
from pneumonia.transforms import preprocess_xray

from pathlib import Path
import torch.nn as nn
from pneumonia.config import LABEL_NAMES

def predict(model: nn.Module, source: Path | BinaryIO) -> dict:
    """Fait une prédiction sur une image donnée, en utilisant un modèle fine-tuné.

    Args:"""


    img = skimage.io.imread(source)      # charger
    img = preprocess_xray(img)          # prétraiter (numpy -> numpy)
    img = torch.from_numpy(img).float() # convertir en tensor
    img = img.unsqueeze(0)              # ajouter la dimension batch

    model.eval()

    with torch.no_grad():
        output = model(img)
        probabilities = torch.softmax(output, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1)[0].item()
        confidence = probabilities[0, predicted_label].item()


    return {
        "label": predicted_label,
        "class_name": LABEL_NAMES[predicted_label],
        "confidence": confidence,
    }
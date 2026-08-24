import torchxrayvision as xrv
import logging
from pneumonia.model_utils import adapt_model_head
import torch.nn as nn
import torch
from pathlib import Path

from pneumonia.config import DEFAULT_WEIGHTS_NAME, NUM_CLASSES

#Load the model
def load_model(weights=DEFAULT_WEIGHTS_NAME):
    """Charge un modèle DenseNet121 pré-entraîné via TorchXRayVision.

    Args:
        weights: Nom des poids pré-entraînés à charger.

    Returns:
        Le modèle chargé (torch.nn.Module).

    Raises:
        RuntimeError: Si le chargement échoue (poids invalides, réseau...).
    """

    try:
        model=xrv.models.DenseNet(weights=weights)
        return model
    
    except Exception as e:
        logging.error(f"{type(e).__name__}")
        raise RuntimeError("Impossible de charger le modèle")


def load_finetuned_model(checkpoint_path: Path, weights_name: str = DEFAULT_WEIGHTS_NAME, num_classes: int = NUM_CLASSES, device: str = "cpu") -> torch.nn.Module:
    """Reconstruit l'architecture (DenseNet121 pré-entraîné + tête adaptée) puis y injecte les poids fine-tunés sauvegardés."""


    try:
        model= load_model(weights=weights_name)
        model= adapt_model_head(model, num_classes=num_classes)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return model
    
    except Exception as e:
        logging.error(f"{type(e).__name__}")
        raise RuntimeError("Impossible de charger le modèle")
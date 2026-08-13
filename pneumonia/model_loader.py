import torchxrayvision as xrv
import logging

#Load the model
def load_model(weights="densenet121-res224-all"):
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
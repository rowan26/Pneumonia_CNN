import torch.nn as nn
from pneumonia.config import NUM_CLASSES

def adapt_model_head(model, num_classes: int=NUM_CLASSES) -> nn.Module:

    """Remplace la tête (dernière couche) du modèle pour l'adapter au
    nombre de classes de la tâche cible.

    Le corps pré-entraîné (extraction de caractéristiques visuelles) est
    conservé intact ; seule la couche de classification finale, spécifique
    aux 18 pathologies d'origine, est remplacée par une nouvelle couche
    linéaire adaptée au nombre de classes voulu.

    Args:
        model: Le modèle déjà chargé (via load_model), à adapter.
        num_classes: Nombre de classes en sortie (2 pour NORMAL/PNEUMONIA).

    Returns:
        Le même modèle, avec sa tête remplacée.
    """

    model.op_threshs = None
    in_features = model.classifier.in_features

    model.classifier = nn.Linear(in_features=in_features, out_features=num_classes)

    return model
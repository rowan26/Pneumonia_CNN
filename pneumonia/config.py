"""Constantes partagées du projet.

Ce module ne contient aucune logique et ne dépend d'aucun autre module :
il peut être importé de partout — préparation des données, entraînement,
inférence — sans entraîner de dépendances inutiles.
"""

# Correspondance label numérique -> nom de classe.
# Convention fixée en Phase 2 : PNEUMONIA est la classe positive (1), ce qui
# aligne le recall calculé par défaut sur la métrique cliniquement prioritaire.
LABEL_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}

# Extensions d'image acceptées.
DEFAULT_ALLOWED_EXTENSIONS = (".jpeg", ".jpg", ".png")

# Poids pré-entraînés TorchXRayVision utilisés comme base.
DEFAULT_WEIGHTS_NAME = "densenet121-res224-all"

# Nombre de classes de la tâche (NORMAL / PNEUMONIA).
NUM_CLASSES = 2
# Pneumonia Detection

Détection de pneumonie à partir de radiographies thoraciques, construite à partir d'un modèle DenseNet121 pré-entraîné ([TorchXRayVision](https://github.com/mlmed/torchxrayvision)) plutôt qu'un CNN entraîné from scratch.

## Contexte

Ce projet vise à construire une pipeline complète et reproductible, du modèle pré-entraîné jusqu'au déploiement, en appliquant des principes MLOps (versioning, reproductibilité, séparation des étapes) et de security by design (validation des entrées, gestion des secrets, absence de données sensibles versionnées) tout au long du développement.

## Stack

- **Modèle** : DenseNet121 pré-entraîné via TorchXRayVision (poids `densenet121-res224-all`), déjà entraîné sur plusieurs centaines de milliers de radiographies thoraciques réelles (ChestX-ray14, CheXpert, PadChest...).
- **Dataset** : [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Paul Mooney, Kaggle) — 5 863 images réparties en NORMAL / PNEUMONIA.
- **Framework** : PyTorch

## Structure du projet

```
.
├── pneumonia/              # Code source importable (modèle, validation, utilitaires dataset)
│   ├── model_loader.py     # Chargement du modèle pré-entraîné
│   ├── input_validation.py # Validation des fichiers image (extension, intégrité)
│   └── dataset_utils.py    # Comptage et récapitulatif du dataset
├── scripts/                # Points d'entrée exécutables
│   └── explore_dataset.py  # Exploration et comptage du dataset
├── data/                   # Dataset (non versionné, voir installation ci-dessous)
└── requirements.txt
```

## Installation

### 1. Environnement Python

```bash
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset

Le dataset n'est pas versionné dans ce repo (volume trop important pour Git). Pour le récupérer :

1. Télécharger [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) depuis Kaggle.
2. Extraire l'archive et placer son contenu de façon à obtenir la structure suivante à la racine du projet :

```
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Utilisation

Récapitulatif du nombre d'images par split et par classe :

```bash
python scripts/explore_dataset.py
```

## Sécurité et fiabilité

- Le dataset n'est jamais versionné dans le repo (voir `.gitignore`).
- Les fichiers image sont validés avant traitement, à deux niveaux : extension attendue, puis intégrité réelle du contenu (`pneumonia/input_validation.py`) — un fichier corrompu ou renommé de façon trompeuse est écarté plutôt que de faire échouer silencieusement le traitement.
- Les erreurs internes ne sont jamais exposées en détail dans les logs (seul le type d'exception est consigné), afin d'éviter la fuite d'informations techniques sensibles.

## License

Le dataset original est distribué par Paul Mooney sur Kaggle sous licence CC BY 4.0.
# Pneumonia Detection — pipeline MLOps de bout en bout

Détection de pneumonie sur radiographies thoraciques, à partir d'un
DenseNet121 pré-entraîné ([TorchXRayVision](https://github.com/mlmed/torchxrayvision))
fine-tuné pour une classification binaire.

L'objectif n'est pas d'atteindre le meilleur score possible, mais de
construire une chaîne complète — données, entraînement, évaluation,
robustesse, packaging, interface — dont **chaque décision est justifiable**.

---

## Résultats

Évaluation sur le **test set** (624 images, jamais utilisées ni pour
l'entraînement ni pour la sélection du checkpoint) :

| Condition | Recall | Precision | F1 | AUC-ROC | Cas ratés | Fausses alertes |
|---|---|---|---|---|---|---|
| Images d'origine | 0,9974 | 0,8294 | 0,9057 | 0,9562 | 1 / 390 | 80 / 234 |
| Bruit gaussien (σ=10) | 0,9974 | 0,8330 | 0,9078 | 0,9732 | 1 | 78 |
| Luminosité ×1,2 | 0,9949 | 0,8380 | 0,9097 | 0,9538 | 2 | 75 |

**Lecture clinique** : le modèle rate 1 cas de pneumonie sur 390. Le recall
est la métrique prioritaire — un faux négatif signifie qu'un patient malade
repart sans traitement, alors qu'un faux positif ne coûte qu'une relecture.

**Limite assumée** : 34 % des patients sains sont signalés à tort. Le
modèle a un biais marqué vers PNEUMONIA, cohérent avec la priorité donnée
au recall, mais coûteux en charge de relecture.

---

## Le finding le plus intéressant du projet

Le test de robustesse a révélé une **asymétrie** que les métriques
classiques ne montraient pas :

| Modèle initial | Cas ratés | AUC |
|---|---|---|
| Images d'origine | 1 | 0,9755 |
| **Luminosité ×1,2** | **22** | **0,9377** |

Une variation d'exposition de 20 % — imperceptible à l'œil, et parfaitement
plausible d'un appareil à l'autre — faisait rater 21 cas supplémentaires.

**Explication** : le bruit gaussien est aléatoire et centré, les structures
anatomiques y survivent. La luminosité est un décalage *systématique* qui
écrase les contrastes dans les zones claires — précisément là où se lit
l'opacité pulmonaire. La chute de l'AUC confirmait qu'il ne s'agissait pas
d'un problème de seuil de décision, mais d'une perte réelle de
discrimination.

**Correction** : data augmentation sur la luminosité (facteur aléatoire
centré sur 1,0, appliqué au train uniquement), puis réentraînement. Les cas
ratés sous perturbation sont retombés de 22 à 2.

**Coût mesuré** : l'AUC sur images d'origine est passée de 0,9755 à 0,9562.
En apprenant à ignorer la luminosité, le modèle a renoncé à des nuances de
contraste qui portaient un peu d'information utile.

**Arbitrage retenu** : des performances stables quelle que soit la
condition (AUC 0,954–0,973) valent mieux qu'un pic conditionné à des images
parfaites, pour un système destiné à recevoir des images de sources
variées.

---

## Architecture

```
.
├── pneumonia/                  # Code importable
│   ├── config.py               # Constantes partagées, sans dépendance
│   ├── input_validation.py     # Validation extension + intégrité
│   ├── dataset_utils.py        # Comptages, split stratifié, poids de classe
│   ├── dataset.py              # Dataset PyTorch + DataLoader
│   ├── transforms.py           # Prétraitement + augmentation
│   ├── perturbations.py        # Perturbations pour le test de robustesse
│   ├── model_loader.py         # Chargement pré-entraîné et fine-tuné
│   ├── model_utils.py          # Remplacement de la tête (18 → 2 sorties)
│   ├── training.py             # Boucle d'entraînement, best checkpoint
│   ├── evaluation.py           # Collecte des prédictions, métriques
│   └── predict.py              # Inférence sur une image
├── scripts/                    # Points d'entrée exécutables
│   ├── explore_dataset.py
│   ├── check_*.py              # Vérifications brique par brique
│   ├── trains.py               # Entraînement + tracking MLflow
│   ├── eval.py                 # Évaluation sur le test set
│   ├── eval_robustness.py      # Comparaison sous perturbations
│   └── predictions.py          # Inférence en ligne de commande
├── streamlit_app/
│   └── app.py                  # Interface d'upload et prédiction
├── Dockerfile
├── requirements.txt            # Développement et entraînement
├── requirements-inference.txt  # Inférence seule, versions figées
└── pyproject.toml
```

Trois catégories séparées dès le départ : ce qu'on **importe**
(`pneumonia/`), ce qu'on **exécute** (`scripts/`), ce qu'on **sert**
(`streamlit_app/`). Le conteneur ne contient que la troisième et les
modules dont elle dépend.

---

## Choix techniques et pourquoi

**Split stratifié en mémoire, seed fixée.** Le split ne déplace aucun
fichier sur le disque : c'est une fonction pure (mêmes entrées + même seed
= même sortie), recréable à l'identique dans n'importe quel environnement à
partir du code seul. Stratifié par classe, pour que les métriques de
validation restent représentatives des proportions réelles (25,7 % NORMAL).

**Pondération de la loss plutôt qu'un sampler** pour le déséquilibre de
classes — ne touche ni au Dataset ni au DataLoader, et n'implique pas de
voir certaines images plusieurs fois par epoch.

**Sauvegarde du meilleur checkpoint, pas du dernier.** `val_loss` atteint
son minimum à l'epoch 4-5 sur les cinq runs réalisés ; au-delà,
`train_loss` continue de descendre vers 0,02 pendant que `val_loss`
remonte. Le modèle retenu est celui qui généralise le mieux.

**Test set intact jusqu'à la fin.** La validation a servi à choisir le
checkpoint, elle est donc optimiste par construction : 97,7 % d'accuracy en
validation contre 87,8 % en test. Seul le test set donne une mesure
impartiale.

**Tracking MLflow** : 11 paramètres et 4 métriques par epoch, backend
SQLite. La traçabilité modèle ↔ run passe par le nom du checkpoint
(`best_model_<8 premiers caractères du run ID>.pth`) plutôt que par le
stockage d'artefacts MLflow, incompatible avec un flux d'entraînement
distant sans complexité disproportionnée.

**Entraînement sur GPU distant.** Mesuré en local : ~50 s/batch, soit ~4 h
par epoch — impraticable. Sur GPU : ~4 min par epoch, facteur 50. Le code
détecte le device automatiquement et tourne à l'identique dans les deux
environnements.

---

## Sécurité

- **Aucune donnée versionnée** : dataset, checkpoints et bases MLflow sont
  exclus de Git. Le dépôt contient du code, pas des données de santé.
- **Validation en deux temps** des fichiers image : extension d'abord
  (filtre rapide), intégrité réelle du contenu ensuite (`PIL.verify()`) —
  une extension `.jpg` ne garantit rien sur le contenu.
- **Aucune écriture disque de l'image uploadée.** L'app traite un flux
  binaire en mémoire. En contexte médical, écrire une radiographie
  identifiable sur un serveur, même temporairement, soulève des obligations
  de conservation, de nettoyage et de traçabilité — évitées ici par
  construction.
- **Logs sans détail d'exception brut** : seul le type d'erreur est
  consigné, pour éviter la fuite de chemins ou d'informations internes.
- **Conteneur non-root**, périmètre de fichiers explicite dans le
  Dockerfile, `pip` retiré de l'image finale.

### Analyse de vulnérabilités

`docker scout cves` sur l'image d'inférence :

| | Avant durcissement | Après |
|---|---|---|
| Total | 49 | **36** |
| MEDIUM | 11 | **4** |
| Paquets vulnérables | 12 | **11** |

Corrections appliquées : mise à jour des paquets système Debian, et
suppression de `pip` de l'image finale — le conteneur n'installe rien à
l'exécution, et sa présence permettrait à un attaquant ayant obtenu
l'exécution de code d'installer des outils supplémentaires.

**CVE restantes, assumées** : les 2 CRITICAL et 2 HIGH concernent `perl`,
présent dans l'image Debian de base, jamais utilisé par l'application, et
sans correctif publié (`Fixed version: not fixed`). Aucune vulnérabilité
n'affecte PyTorch, Pillow, numpy ou scikit-image — les paquets qui traitent
réellement les données entrantes.

*Piste identifiée* : une image distroless éliminerait la quasi-totalité des
CVE restantes, au prix d'un multi-stage build et d'un débogage sans shell.

---

## Optimisation de l'image Docker

| | Première version | Version actuelle |
|---|---|---|
| Taille disque | 8,56 GB | **1,99 GB** |
| Transfert réseau | 3,03 GB | **480 MB** |

La cause du surpoids initial : `pip install torch` récupère par défaut la
variante **CUDA**, avec tous les pilotes GPU — alors que l'inférence tourne
sur CPU en une à deux secondes. Bascule sur l'index CPU-only de PyTorch.

Deux autres points traités :
- **Poids TorchXRayVision pré-téléchargés au build** — sans cela, chaque
  démarrage retéléchargeait 300 Mo depuis GitHub : dépendance réseau à
  l'exécution et point de défaillance externe pour un outil de dépistage.
- **`.dockerignore`** — contexte de build réduit de plus d'1 Go à 28 Mo.

---

## Installation

### 1. Environnement

```bash
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Python 3.12 requis (le code utilise la syntaxe `Path | None`, disponible
depuis 3.10).

### 2. Dataset

Le dataset n'est pas versionné. Télécharger
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
et l'extraire pour obtenir :

```
data/chest_xray/
├── train/{NORMAL,PNEUMONIA}/
├── val/{NORMAL,PNEUMONIA}/
└── test/{NORMAL,PNEUMONIA}/
```

### 3. Modèle entraîné

Le checkpoint (~22 Mo) n'est pas versionné non plus. Il se régénère avec
`python scripts/trains.py` et doit être placé dans
`artifacts/best_model.pth`.

---

## Utilisation

```bash
# Exploration du dataset
python scripts/explore_dataset.py

# Vérification du split et des poids de classe
python scripts/check_split.py

# Entraînement (tracking MLflow inclus)
python scripts/trains.py

# Évaluation sur le test set
python scripts/eval.py

# Test de robustesse
python scripts/eval_robustness.py

# Prédiction sur une image
python scripts/predictions.py chemin/vers/image.jpeg

# Interface web
streamlit run streamlit_app/app.py
```

### Consulter les expériences MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Conteneur

```bash
docker build -t pneumonia-cnn .
docker run --rm -p 8501:8501 pneumonia-cnn
```

Interface accessible sur `http://localhost:8501`.

---

## Limites

Ce projet est une démonstration technique, **pas un dispositif médical
validé**. En particulier :

- 34 % des patients sains sont signalés à tort — inacceptable pour un usage
  clinique sans relecture systématique.
- Le modèle n'a été évalué que sur le dataset Kaggle, dont les images
  proviennent d'un seul centre. Rien ne garantit ses performances sur
  d'autres populations ou d'autres appareils.
- Le test de robustesse ne couvre que deux types de perturbations. D'autres
  variations réelles (positionnement du patient, artefacts, pathologies
  concomitantes) n'ont pas été testées.
- Aucune validation clinique, aucune revue par un radiologue.

---

## Prochaines étapes

- Déploiement de l'interface sur une plateforme accessible publiquement
- Explicabilité (Grad-CAM) — visualiser les zones qui ont influencé la
  décision, tant pour la confiance utilisateur que pour vérifier que le
  modèle regarde bien les poumons et non un artefact d'image
- Tests unitaires (`pytest`)
- Monitoring en conditions réelles et documentation de conformité
  (logique EU AI Act / MDR)

---

## Licence

Le dataset est distribué par Paul Mooney sur Kaggle sous licence CC BY 4.0.

from typing import Callable

import torchxrayvision as xrv
import numpy as np


def preprocess_xray(img: np.ndarray, normalize_max: int = 255, resize_to: int =224) -> np.ndarray :

    """Applique la chaîne de prétraitement TorchXRayVision : normalisation,
    conversion en niveaux de gris, crop centré, puis redimensionnement.
    """

    img=xrv.datasets.normalize(img,normalize_max)
    if img.ndim==3:
        img=img.mean(2)[None,...]
    elif img.ndim==2:
            img=img[None,...]
    else:
         raise ValueError(f"Nombre de dimensions inattendu pour l'image : {img.ndim}")
    img=xrv.datasets.XRayCenterCrop()(img)
    img=xrv.datasets.XRayResizer(resize_to)(img)
    return img


def random_luminance(img: np.ndarray, low: float=0.8, high: float=1.2, rng=None) -> np.ndarray:
    """Applique un facteur de luminosité tiré au hasard, pour l'augmentation
    de données à l'entraînement.

    Contrairement à `apply_luminance_scaling` (facteur fixe, utilisé pour
    tester la robustesse), le facteur varie à chaque appel : c'est cette
    variété qui apprend au modèle que la luminosité n'est pas un signal
    pertinent. La plage est centrée sur 1.0 pour que le modèle voie autant
    d'images assombries qu'éclaircies, sans décaler la distribution qu'il
    rencontrera en production.

    À n'utiliser que sur les données d'entraînement : appliquée à la
    validation, elle rendrait les métriques instables d'une epoch à l'autre
    et fausserait la sélection du meilleur checkpoint.

    Args:
        img: Image brute (0-255).
        low: Facteur minimum (< 1 assombrit).
        high: Facteur maximum (> 1 éclaircit).
        rng: Générateur numpy optionnel, pour un entraînement reproductible.
    """
    
    if rng is None:
        rng = np.random.default_rng()
    scaler = rng.uniform(low, high)
    scaled_img = img.astype(np.float32) * scaler
    return np.clip(scaled_img, 0, 255).astype(np.uint8)


def compose_with_preprocess(perturbation: Callable) -> Callable:
    """Combine une perturbation et le prétraitement normal en un seul
    callable, utilisable directement comme `transform` d'un ChestXrayDataset.

    La perturbation s'applique sur l'image brute, avant preprocess_xray :
    c'est l'ordre réel en conditions cliniques, où l'image arrive déjà
    dégradée et le prétraitement s'applique ensuite normalement.
    """

    
    def transform(img):
        img = perturbation(img)
        return preprocess_xray(img)
    return transform
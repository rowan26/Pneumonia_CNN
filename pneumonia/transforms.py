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
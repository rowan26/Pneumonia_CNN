import numpy as np



def add_gaussian_noise(img: np.ndarray, sigma: float = 10.0, seed: int | None = None) -> np.ndarray:
    """Ajoute un bruit gaussien centré à l'image, pour simuler du bruit de
    capteur ou une numérisation de mauvaise qualité.

    Les valeurs sont écrêtées entre 0 et 255 : sans cela, la conversion en
    uint8 provoquerait un débordement circulaire (un pixel à 270 deviendrait
    14, créant des artefacts sombres au lieu d'un bruit léger).

    Args:
        img: Image brute (0-255).
        sigma: Écart-type du bruit ; plus il est élevé, plus la dégradation
            est forte (10 ≈ 4 % de l'échelle).
        seed: Graine pour rendre la perturbation reproductible.
    """


    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    noisy_img = img.astype(np.float32) + noise

    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def apply_luminance_scaling(img: np.ndarray, scaler: float = 1.2) -> np.ndarray:
    """Multiplie l'intensité des pixels, pour simuler un appareil ou un
    réglage d'exposition différent.

    La multiplication (plutôt qu'une addition constante) reproduit le
    comportement physique d'une variation d'exposition : les zones claires
    s'éclaircissent proportionnellement plus que les zones sombres.

    Args:
        img: Image brute (0-255).
        scaler: Facteur multiplicatif (>1 éclaircit, <1 assombrit).
    """


    scaled_img = img.astype(np.float32) * scaler
    return np.clip(scaled_img, 0, 255).astype(np.uint8)
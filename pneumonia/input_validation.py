DEFAULT_ALLOWED_EXTENSIONS = (".jpeg", ".jpg", ".png")

from PIL import Image
from pathlib import Path


def is_valid_extension(filename: str, allowed_extensions: tuple[str, ...] = DEFAULT_ALLOWED_EXTENSIONS) -> bool:
    """Vérifie si un nom de fichier a une extension autorisée (insensible à la casse)."""
    
    return filename.lower().endswith(allowed_extensions)


def is_valid_image(path: Path) -> bool:
    """Vérifie qu'un fichier est une image lisible et non corrompue.

    N'ouvre le fichier que pour le vérifier ; ne modifie ni ne supprime rien.
    """
    try:
        Image.open(path).verify()
        return True
    except (IOError, SyntaxError): #IOError when programming exception occurs when a system taks failed
        return False
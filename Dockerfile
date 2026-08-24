FROM python:3.12-slim

WORKDIR /app

# Mise à jour des paquets système Debian : corrige les CVE pour lesquelles
# un correctif existe (util-linux notamment). Le nettoyage des listes apt
# dans la même instruction évite de les laisser dans la couche finale.
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# pip est retiré après installation : le conteneur n'installe rien à
# l'exécution, et sa présence permettrait à un attaquant ayant obtenu
# l'exécution de code d'installer des outils supplémentaires.
RUN pip uninstall -y pip \
    && rm -rf /usr/local/lib/python3.12/site-packages/pip*

USER appuser

RUN python -c "import torchxrayvision as xrv; xrv.models.DenseNet(weights='densenet121-res224-all')"

COPY --chown=appuser:appuser pneumonia/__init__.py pneumonia/
COPY --chown=appuser:appuser pneumonia/config.py pneumonia/
COPY --chown=appuser:appuser pneumonia/input_validation.py pneumonia/
COPY --chown=appuser:appuser pneumonia/model_loader.py pneumonia/
COPY --chown=appuser:appuser pneumonia/model_utils.py pneumonia/
COPY --chown=appuser:appuser pneumonia/transforms.py pneumonia/
COPY --chown=appuser:appuser pneumonia/predict.py pneumonia/

COPY --chown=appuser:appuser scripts/__init__.py scripts/
COPY --chown=appuser:appuser scripts/predictions.py scripts/

COPY --chown=appuser:appuser artifacts/best_model.pth artifacts/

ENTRYPOINT ["python", "-m", "scripts.predictions"]
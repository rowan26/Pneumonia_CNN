FROM python:3.12-slim

WORKDIR /app

# Mise à jour des paquets système Debian : corrige les CVE pour lesquelles
# un correctif existe. Le nettoyage des listes apt dans la même instruction
# évite de les laisser dans la couche finale.
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

# Dépendances d'abord : cette couche reste en cache tant que
# requirements-inference.txt ne change pas.
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# Code d'inférence uniquement. Liste explicite plutôt qu'un COPY du dossier
# entier : elle documente le périmètre, et le build échoue si un module
# d'entraînement venait à être importé par erreur.
COPY pneumonia/__init__.py pneumonia/
COPY pneumonia/config.py pneumonia/
COPY pneumonia/input_validation.py pneumonia/
COPY pneumonia/model_loader.py pneumonia/
COPY pneumonia/model_utils.py pneumonia/
COPY pneumonia/transforms.py pneumonia/
COPY pneumonia/predict.py pneumonia/

COPY scripts/__init__.py scripts/
COPY scripts/predictions.py scripts/

COPY streamlit_app/app.py streamlit_app/

COPY artifacts/best_model.pth artifacts/

COPY pyproject.toml .

# Enregistre le package pour que les imports fonctionnent quel que soit
# le répertoire depuis lequel l'application est lancée.
RUN pip install --no-cache-dir -e .

# pip retiré une fois toutes les installations terminées : le conteneur
# n'installe rien à l'exécution, et sa présence permettrait à un attaquant
# ayant obtenu l'exécution de code d'installer des outils supplémentaires.
RUN pip uninstall -y pip \
    && rm -rf /usr/local/lib/python3.12/site-packages/pip*

RUN chown -R appuser:appuser /app
USER appuser

# Poids TorchXRayVision pré-téléchargés au build, sous appuser pour que le
# cache aille dans /home/appuser. Sans cela, chaque démarrage de conteneur
# retéléchargerait environ 300 Mo depuis GitHub : dépendance réseau à
# l'exécution et point de défaillance externe.
RUN python -c "import torchxrayvision as xrv; xrv.models.DenseNet(weights='densenet121-res224-all')"

# Cloud Run impose que l'application écoute sur le port fourni par la
# variable PORT. La forme shell de CMD est nécessaire pour que $PORT soit
# résolu au démarrage (la forme exec ne l'interpréterait pas).
ENV PORT=8080
EXPOSE 8080

# CORS et XSRF désactivés : le proxy de Cloud Run réécrit les en-têtes
# d'origine, ce qui fait échouer les vérifications de Streamlit sur des
# requêtes pourtant légitimes, notamment l'upload de fichier. Acceptable
# ici — application publique, sans authentification ni données persistées.
# À ne pas reproduire sur une application gérant des sessions utilisateur.
CMD streamlit run streamlit_app/app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# Code d'inférence
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

# pip retiré une fois toutes les installations terminées.
RUN pip uninstall -y pip \
    && rm -rf /usr/local/lib/python3.12/site-packages/pip*

RUN chown -R appuser:appuser /app
USER appuser

RUN python -c "import torchxrayvision as xrv; xrv.models.DenseNet(weights='densenet121-res224-all')"

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
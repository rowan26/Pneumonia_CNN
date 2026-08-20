from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import torch


def collect_predictions(model, dataloader, device) -> tuple[list[int], list[int], list[float]] :
    """Parcourt un dataloader et collecte les vrais labels, les prédictions
    et les probabilités de la classe PNEUMONIA.

    Les probabilités (softmax sur la classe 1) sont conservées en plus des
    prédictions binaires car l'AUC-ROC se calcule sur les scores de
    confiance, pas sur la décision finale.

    Returns:
        Un tuple (labels, prédictions, probabilités), une entrée par image.
    """


    model.eval()
    all_labels = []
    all_preds = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    return all_labels, all_preds, all_probabilities


def compute_metrics(labels: list[int], preds: list[int], probabilities: list[float]) -> dict:
    """Calcule les métriques d'évaluation pour la classification binaire.

    La classe positive est PNEUMONIA (label 1), ce qui fait du recall la
    proportion de cas de pneumonie effectivement détectés — la métrique
    cliniquement prioritaire ici.

    Args:
        labels: Vrais labels.
        preds: Prédictions binaires du modèle.
        probabilities: Probabilités de la classe PNEUMONIA, pour l'AUC-ROC.

    Returns:
        Un dictionnaire avec precision, recall, F1, AUC-ROC et les quatre
        cases de la matrice de confusion.
    """

    
    precision=precision_score(labels, preds, average='binary')
    recall=recall_score(labels, preds, average='binary')
    f1=f1_score(labels, preds, average='binary')
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    roc_auc=roc_auc_score(labels, probabilities)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }